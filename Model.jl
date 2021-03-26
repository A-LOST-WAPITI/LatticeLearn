using Flux, Statistics, Random, BSON, Dates, CUDA
using Flux: train!, Optimiser
using Flux.Losses: mse
using Parameters: @with_kw
using Base.Iterators: partition
using LinearAlgebra
using Printf
using Plots; pyplot()


using DataProduce


# Metaparameters
@with_kw mutable struct _Args
    DataSource::String = "Rect"         # source of data
    nodeNum::Int = 32                   # node
    Supercell::NTuple{2, Int} = (1, 1)  # supercell
    η::Float64 = 3e-4                   # learning rate
    batchSize::Int = 32                 # batch size
    epochs::Int = 10                    # number of epochs
    gpuFlag::Bool = false               # use gpu or not
    decayStep::Int = 100                # how many steps to decay
    decayRatio::Float64 = 0.8           # decay ratio
    ηClip::Float64 = 1e-6               # least learning rate
    noiseRatio::Int = 50                # noise times over learning rate
end


# Get data
function _GetData(Args::_Args)
    InData, TargetData, FormulaArray = _DataProduce(
        DataSource = Args.DataSource,
        nodeNum = Args.nodeNum, 
        Supercell = Args.Supercell
    )

    materialCount = size(TargetData, 2)
    ShuffleIndexArray = randperm(materialCount)
    InData = InData[:, :, :, ShuffleIndexArray]
    TargetData = TargetData[:, ShuffleIndexArray]
    FormulaArray = FormulaArray[ShuffleIndexArray, :]
    cutOff = materialCount * 0.7 |> round |> Int64
    TrainX = InData[:, :, :, 1:cutOff]
    TrainY = TargetData[:, 1:cutOff]
    TrainFormulaArray = FormulaArray[1:cutOff]
    TestX = InData[:, :, :, cutOff + 1:end]
    TestY = TargetData[:, cutOff + 1:end]
    TestFormulaArray = FormulaArray[cutOff + 1:end]

    TrainBatchIndicsArray = partition(1:cutOff, Args.batchSize)
    TrainData = [
        (
            TrainX[:, :, :, Indics],
            TrainY[:, Indics]
        ) for Indics in TrainBatchIndicsArray
    ]
    TestData = [(TestX, TestY)]

    return TrainData, TestData, TrainFormulaArray, TestFormulaArray, size(InData), size(TargetData, 1)
end


# Build model
function _BuildModel(
    InSize::NTuple{N, Int},
    nOut::Int
) where N
    nodeNum = InSize[1]
    nChannel = InSize[3]
    nodeNumAfterPool = nodeNum ÷ 8

    Model = Chain(
        # Fist convolution
        Conv((3, 3), nChannel => 32, pad = (1, 1), gelu),
        MeanPool((2, 2)),

        # Second convolution
        # Conv((3, 3), 32 => 32, pad = (1, 1), gelu),
        # MeanPool((2, 2)),

        # Third convolution
        Conv((3, 3), 32 => 8, pad = (1, 1), gelu),
        MeanPool((2, 2)),

        # Fourth convolution
        Conv((3, 3), 8 => 4, pad = (1, 1), gelu),
        MeanPool((2, 2)),

        # Dense
        (X -> flatten(X)),
        # Dropout(0.2),
        Dense(nodeNumAfterPool^2 * 4, nOut^2, gelu),
        # Dropout(0.2),
        Dense(nOut^2, nOut)
    )

    return Model
end


# Loss and accuracy
function _Val(
    Data::AbstractVector{
        <:Tuple{
            AbstractArray{T},
            AbstractArray{T}
        }
    },
    Model::Chain
) where T
    AllX = cat(
        (Data .|> (X -> X[1]))...,
        dims = length(size(Data[1][1]))
    )
    AllY = cat(
        (Data .|> (X -> X[2]))...,
        dims = length(size(Data[1][2]))
    )

    AllPredict = Model(AllX)

    loss = mse(AllPredict, AllY)

    NonZeroIndics = AllY .|> (x -> isapprox(x, 0, atol = 1e-1)) .|> !
    err = mean((@. 1 - AllPredict[NonZeroIndics] / AllY[NonZeroIndics]) .|> abs)

    MeanArray = repeat(
        mean(AllY, dims = 2),
        outer = (1, size(AllY, 2))
    )

    UpperTempArray = sum((@. (AllPredict - AllY)^2), dims = 2)
    LowerTempArray = sum((@. (MeanArray - AllY)^2), dims = 2)

    R2Array = @. 1 - UpperTempArray / LowerTempArray
    R2Array = R2Array[R2Array .|> isinf .|> !]

    p = size(AllX)[end - 1]
    n = size(AllX)[end]
    r2Adj = mean(@. 1 - (1 - R2Array) * (n - 1) / (n - p - 1))

    return loss, err, r2Adj
end


# Train
function _Train(; kws...)
    # Initializing model parameters
    Args = _Args(; kws...)

    Args.gpuFlag = CUDA.has_cuda()

    # Load Data
    @info "Loading data set..."
    (
        TrainData, 
        TestData, 
        TrainFormulaArray, 
        TestFormulaArray,
        InSize,
        nOut
    ) = _GetData(Args)

    # Build model
    @info "Building model..."
    Model = _BuildModel(InSize, nOut)
    _Loss(X, Y) = mse(Model(X), Y)

    # Move data and model to gpu
    if Args.gpuFlag
        @info "CUDA is on"
        CUDA.allowscalar(false)

        TrainData = TrainData .|> gpu
        TestData = TestData .|> gpu

        Model = Model |> gpu
    end

    # Training
    # opt = AMSGrad(Args.η, (0.89, 0.995))
    opt = ADAMW(Args.η, (0.89, 0.995), 0.0001)

    @info "Beginning training loop..."
    @printf(
        "[epoch index]:%15s\t%15s\t%15s\t%15s\t%15s\t%15s\n",
        " |TrainLoss|",
        "  |TestLoss|",
        "|TrainError|",
        " |TestError|",
        "   |TrainR2|",
        "    |TestR2|"
    )
    leastError = Inf64
    lastImprovement = 0
    TrainLossArray = Array{Float64}(undef, Args.epochs)
    TestLossArray = Array{Float64}(undef, Args.epochs)
    for epochIndex = 1:Args.epochs
        trainmode!(Model, true)
        train!(
            _Loss,
            params(Model),
            TrainData,
            opt
        )

        testmode!(Model, true)
        trainLoss, trainErr, trainR2 = _Val(TrainData, Model)
        testLoss, testErr, testR2 = _Val(TestData, Model)
        TrainLossArray[epochIndex] = trainLoss
        TestLossArray[epochIndex] = testLoss

        @printf(
            "[epoch %5d]:%15.2f\t%15.2f\t%15.2f\t%15.2f\t%15.2f\t%15.2f\n",
            epochIndex, 
            trainLoss, 
            testLoss, 
            trainErr,
            testErr,
            trainR2,
            testR2
        )
    end

    # download data and model to cpu
    if Args.gpuFlag
        TrainData = TrainData .|> cpu
        TestData = TestData .|> cpu

        Model = Model |> cpu
    end

    Xs = Array{Float64}(undef, 0, 6)
    Ys = Array{Float64}(undef, 0, 6)
    for (X, Y) in TestData
        Xs = vcat(Xs, Y')
        Ys = vcat(Ys, Model(X)')
    end

    testMin, testMax = extrema(Xs)
    predictMin, predictMax = extrema(Ys)
    plotStart = min(testMin, predictMin)
    plotEnd = max(testMax, predictMax)

    plt = plot(
        dpi = 400,
        background_color = :transparent,
        legend = :outertopright
    )
    Plots.scatter!(
        plt,
        Xs,
        Ys,
        lab = ["c11" "c12" "c13" "c22" "c23" "c33"]
    )
    Plots.plot!(
        plt,
        plotStart:plotEnd,
        (x -> x),
        label = nothing
    )
    savefig(plt, "test.svg")

    plt = plot(
        dpi = 400,
        background_color = :transparent,
        legend = :outertopright
    )
    Plots.scatter!(
        plt,
        Xs,
        Ys,
        lab = ["c11" "c12" "c13" "c22" "c23" "c33"],
        series_annotations = TrainFormulaArray .|> (X -> Plots.text(X, :black, :right, 5))
    )
    Plots.plot!(
        plt,
        plotStart:plotEnd,
        (x -> x),
        label = nothing
    )
    savefig(plt, "testWithFormula.png")

    Xs = Array{Float64}(undef, 0, 6)
    Ys = Array{Float64}(undef, 0, 6)
    for (X, Y) in TrainData
        Xs = vcat(Xs, Y')
        Ys = vcat(Ys, Model(X)')
    end

    testMin, testMax = extrema(Xs)
    predictMin, predictMax = extrema(Ys)
    plotStart = min(testMin, predictMin)
    plotEnd = max(testMax, predictMax)

    plt = plot(
        dpi = 400,
        background_color = :transparent,
        legend = :outertopright
    )
    Plots.scatter!(
        plt,
        Xs,
        Ys,
        lab = ["c11" "c12" "c13" "c22" "c23" "c33"]
    )
    plot!(
        plt,
        plotStart:plotEnd,
        (x -> x),
        label = nothing
    )
    savefig(plt, "train.png")

    plt = plot(
        dpi = 400,
        background_color = :transparent,
        legend = :outertopright
    )
    plot!(
        TrainLossArray,
        label = "Train"
    )
    plot!(
        TestLossArray,
        label = "Test"
    )
    savefig(plt, "curvePreBatch.png")

    plt = plot(
        dpi = 400,
        background_color = :transparent,
        legend = :outertopright
    )
    plot!(
        TrainLossArray,
        label = "Train"
    )
    plot!(
        TestLossArray,
        label = "Test"
    )
    savefig(plt, "curvePreEpoch.png")

    BSON.@save "$(Dates.format(now(), "yyyy-mm-dd#HH-MM")).bson" Model
end