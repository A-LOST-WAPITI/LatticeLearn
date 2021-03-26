"""
用于数据处理的模块。

导出的函数名为`_DataProduce`。
"""
module DataProduce


# 标准库
using DelimitedFiles, LinearAlgebra, Statistics
# 第三方库
try
    using InvertedIndices, JLD
catch error
    using Pkg

    Pkg.add(["InvertedIndices", "JLD"])
    using InvertedIndices, JLD
end


# 自己的模块
using GetData


# 导出的函数
export _DataProduce


"""
    _ZScoreNormalization!(X!::AbstractArray{T}) where T

用于进行Zero-Score标准化的函数。
允许`X!`中存在`nothing`，在计算过程中会被忽略。
"""
function _ZScoreNormalization!(X!::AbstractArray{T}) where T
    NothingIndexArray = findall(isnothing, X!)  # 找到所有的nothing的位置
    NormalX! = view(X!, Not(NothingIndexArray)) # 引用所有非nothing成数组

    X̄ = NormalX! |> mean    # 均值
    σ = NormalX! |> std     # 标准差

    NormalX! .= @. (NormalX! - X̄)/σ # 自身更改
end


"""
    _BuildStiffnessTensor(Cs::AbstractVector{T}) where T

用于将弹性模量组成的Vector转换为Matrix。
"""
function _BuildStiffnessTensor(Cs::AbstractVector{T}) where T
    c11, c12, c13, c22, c23, c33 = Cs

    Stiffness = [
        c11 c12 c13;
        c12 c22 c23;
        c13 c23 c33
    ]

    return Stiffness
end


"""
    _StiffnessStableCheck(Stiffness::AbstractArray{T, 2}) where T

通过计算StiffnessTensor的最小本征值是否大于零来检测材料的稳定性。
"""
function _StiffnessStableCheck(Stiffness::AbstractArray{T, 2}) where T
    stableFlag = (Stiffness |> eigvals |> minimum) > 0

    return stableFlag
end


"""
    _DataProduce(
        ::Type{T};
        nodeNum::Int = 32,
        Supercell::NTuple{2, Int} = (1, 1)
    ) where T

用于处理不同材料制定性质数据的主程序。

# 参数
- `::Type{T}`为返回数据的数据类型；
- `nodeNum::Int = 32`为单方向划分格点的数量；
- `Supercell::NTuple{2, Int} = (1, 1)`为沿
"""
function _DataProduce(
    ::Type{T};
    DataSource::String = "Rect",
    nodeNum::Int = 32,
    Supercell::NTuple{2, Int} = (1, 1)
) where T
    PropertyNameArray = readdlm("ElementProperty.in", ',') |> vec   # 读取所需的元素性质

    # 用于给出当前所选性质的hash标志
    ProducedDataFile = DataSource * "Data-" * (
        [
            PropertyNameArray; 
            nodeNum |> repr; 
            Supercell |> collect .|> repr
        ] |> hash |> repr
    ) * ".jld"

    if isfile(ProducedDataFile) # 如果已经存在符合描述的数据文件
        # 从jld读取数据
        InDataTemp, TargetDataTemp, FormulaArray = load(
            ProducedDataFile,
            "InData",
            "TargetData",
            "FormulaArray"
        )

        # 数据格式
        InData = Array{T}(InDataTemp)
        TargetData = Array{T}(TargetDataTemp)
    else    # 如果没有符合描述的数据文件
        MaterialDF, ElementsDict = _GetData(DataSource)   # 获取材料数据与元素性质数据

        PropertyDict = Dict{String, Dict{String, Union{T, Nothing}}}()  # 用于存储各元素各个性质的Dict
        ElementSymbolArray = ElementsDict |> keys |> collect    # 所有元素名称的数组
        # 元素性质的标准化
        for PropertyName in PropertyNameArray   # 遍历所有元素性质
            # 获取所有元素指定性质的数值
            PropertyValueArray = Union{T, Nothing}[
                ElementsDict[ElementSymbol][PropertyName]
                    for ElementSymbol in ElementSymbolArray
            ]
            _ZScoreNormalization!(PropertyValueArray)   # 标准化

            # 将数组转为Dict
            PropertyDict[PropertyName] = Dict(
                [
                    (ElementSymbolArray[i] => PropertyValueArray[i])
                        for i in eachindex(ElementSymbolArray)
                ]
            )
        end

        FormulaArray = MaterialDF.formula |> Array  # 所有材料的化学式
        TargetData = MaterialDF[:, Not("formula")] |> Array{T} |> permutedims   # 获取除了化学式之外的材料数据（弹性模量数据）
        # 输入数据预分配内存空间
        InData = zeros(
            T, 
            (
                nodeNum * Supercell[1], 
                nodeNum * Supercell[2], 
                length(PropertyNameArray) + 1, 
                size(MaterialDF, 1)
            )
        )
        ErrorMaterialIndexArray = Int[] # 出现错误的数据索引数组
        for (materialIndex, Formula) in enumerate(FormulaArray) # 遍历所有材料的化学式
            if view(TargetData, :, materialIndex) |> _BuildStiffnessTensor |> _StiffnessStableCheck # 如果材料稳定
                PositionTemp = readdlm(DataSource * "Result/" * Formula * "/Position.dat")               # 获取材料原子分数坐标
                LatticeVecArray::Array{T} = readdlm(DataSource * "Result/" * Formula * "/Lattice.dat")   # 获取材料格矢

                ElementSymbolArray = PositionTemp[1, :]             # 获取材料所含元素
                filter!((X -> length(X) > 0), ElementSymbolArray)   # 去除无关数据
                ElementNumArray = PositionTemp[2, :]                # 获取材料各元素数量
                filter!((X -> length(X) > 0), ElementNumArray)      # 去除无关数据
                ElementNumArray = Array{Int}(ElementNumArray)       # 元素数量数组格式化为整形
                # 根据不同元素的数量重复元素名称
                AtomSymbolArray = vcat(
                    [
                        fill(
                            ElementSymbolArray[i],
                            ElementNumArray[i]
                        ) for i in eachindex(ElementNumArray)
                    ]...
                )

                elementNumSum = sum(ElementNumArray)    # 原子总数
                # 计算划分格点下的相对位置
                RelativePositionArray = Array{T}(
                    PositionTemp[4:3 + elementNumSum, 1:3] .|> (x -> x > 1 ? 1 : x)
                ) * (nodeNum - 1) .|> floor .|> Int
                RelativePositionArray = RelativePositionArray .|> (x -> x < 0 ? x + nodeNum : x + 1)

                # 生成超胞
                AtomSymbolArray = repeat(AtomSymbolArray, prod(Supercell))
                RelativePositionArray = vcat(
                    [
                        RelativePositionArray .+ repeat(
                            [(xPeriod - 1) * nodeNum (yPeriod - 1) * nodeNum 0],
                            elementNumSum
                        ) for xPeriod = 1:Supercell[1] for yPeriod = 1:Supercell[2]
                    ]...
                )

                # 计算Bulk的相对长度
                BulkRatioArray = RelativePositionArray[:, 3] ./ (nodeNum - 1)
                BulkRatioArray .-= mean(BulkRatioArray)
                # 填充数据
                for (atomIndex, AtomSymbol) in enumerate(AtomSymbolArray)   # 遍历原子
                    errorFlag = false   # 材料错误的标志
                    xIndex, yIndex, _ = RelativePositionArray[atomIndex, :] # 原子的相对位置

                    # 获取原子的实际位置
                    x, y, z = LatticeVecArray' * [(xIndex - 1)/(nodeNum - 1), (yIndex - 1)/(nodeNum - 1), BulkRatioArray[atomIndex]] .|> T
                    PropertyArray = zeros(T, length(PropertyNameArray)) # 预分配原子性质数组
                    for (propertyIndex, PropertyName) in enumerate(PropertyNameArray)   # 遍历所有元素性质
                        property = PropertyDict[PropertyName][AtomSymbol]   # 获取元素性质

                        if isnothing(property)  # 如果出现nothing则标记为错误
                            errorFlag = true
                            break
                        else    # 若非nothing则记录数据
                            PropertyArray[propertyIndex] = property
                        end
                    end

                    InData[xIndex, yIndex, :, materialIndex] .= [z; PropertyArray]    # 记录数据
                    if errorFlag    # 若标记为错误则记录错误材料的索引
                        push!(ErrorMaterialIndexArray, materialIndex)
                    end
                end
            else    # 若材料不稳定则标记为错误
                push!(ErrorMaterialIndexArray, materialIndex)
            end
        end

        # 排除错误材料
        InData = InData[:, :, :, Not(ErrorMaterialIndexArray)]
        TargetData = TargetData[:, Not(ErrorMaterialIndexArray)]
        FormulaArray = FormulaArray[Not(ErrorMaterialIndexArray)]

        # 旋转数据增强
        InData = cat(
            [
                mapslices((X -> rotr90(X, rotationTime)), InData, dims = [1, 2])
                for rotationTime = 0:1
            ]...,
            dims = 4
        )
        TargetData = cat(
            (
                TargetData,
                TargetData[[4, 2, 5, 1, 3, 6], :]
            )...,
            dims = 2
        )
        FormulaArray = repeat(FormulaArray, outer=2)

        save(
            ProducedDataFile,
            "InData",
            InData,
            "TargetData",
            TargetData,
            "FormulaArray",
            FormulaArray
        )
    end

    return InData, TargetData, FormulaArray
end
_DataProduce(
    ;
    DataSource::String = "Rect",
    nodeNum::Int = 32,
    Supercell::NTuple{2, Int} = (1, 1)
) = _DataProduce(Float32, DataSource = DataSource, nodeNum = nodeNum, Supercell = Supercell)


end
