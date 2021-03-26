"""
用于获取数据的模块。

导出的函数名为`_GetData`
"""
module GetData

# 标准库
using DelimitedFiles, LinearAlgebra
# 第三方库
try
    using JSON, DataFrames, CSV, InvertedIndices
catch error
    using Pkg

    Pkg.add(["JSON", "DataFrames", "CSV", "InvertedIndices"])
end


const ELEMENTS_DATA_PATH = "Data/PeriodicTableJSON.json"
const AIAB_ENERGY_PATH = "Data/element_aiab_energy.json"
const FIXED_ELEMENTS_DATA_PATH = "FixedElementsData.json"
const RECT_DATA_DIR_PATH = "RectResult/"
const C2DB_DATA_DIR_PATH = "C2DBResult/"


# 导出的函数
export _GetData


# 将两个Json文件合并转为Julia的Dict
function _Json2Dict()
    # 查看元素周期表Json是否已就绪，若未就绪则下载
    if !isfile(ELEMENTS_DATA_PATH)
        download(
            "https://github.com/Bowserinator/Periodic-Table-JSON/blob/master/PeriodicTableJSON.json",
            ELEMENTS_DATA_PATH
        )
    end

    Elements = JSON.parsefile(ELEMENTS_DATA_PATH)["elements"]   # 处理元素周期表Json
    ElementsAIAB = JSON.parsefile(AIAB_ENERGY_PATH)             # 处理AIAB的Json
    ElementsDict = Dict{String,Dict{String,Any}}()              # 存储结果的Dict

    for Element in Elements
        ElementSymbol = Element["symbol"]
        
        ElementsDict[ElementSymbol] = Element
        # 有些原子的AIAB Energy缺失用nothing填充
        ElementsDict[ElementSymbol]["aiab"] = haskey(ElementsAIAB, ElementSymbol) ? ElementsAIAB[ElementSymbol][1] : nothing
    end

    return ElementsDict
end


function _GetData(DataSource::String)
    c2dbFlag = false
    if DataSource == "Rect"
        MaterialDataDirPath = RECT_DATA_DIR_PATH
    elseif DataSource == "C2DB"
        MaterialDataDirPath = C2DB_DATA_DIR_PATH
        c2dbFlag = true
    end

    if FIXED_ELEMENTS_DATA_PATH |> isfile
        ElementsDict = JSON.parsefile(FIXED_ELEMENTS_DATA_PATH)
    else
        ElementsDict = _Json2Dict()

        open(FIXED_ELEMENTS_DATA_PATH, "w") do io
            write(
                io,
                JSON.json(ElementsDict)
            )
        end
    end

    FixedMaterialDataPath = "Fixed" * DataSource * "MaterialData.csv"
    if FixedMaterialDataPath |> isfile
        MaterialDF = CSV.read(FixedMaterialDataPath, DataFrame)
    else
        MaterialFormulaArray = readdir(MaterialDataDirPath)

        C11Array = similar(MaterialFormulaArray, Float64)
        C12Array = similar(MaterialFormulaArray, Float64)
        C13Array = similar(MaterialFormulaArray, Float64)
        C22Array = similar(MaterialFormulaArray, Float64)
        C23Array = similar(MaterialFormulaArray, Float64)
        C33Array = similar(MaterialFormulaArray, Float64)

        ErrorIndics = Int64[]
        for (index, MaterialFormula) in enumerate(MaterialFormulaArray)
            area::Float64 = 0
            StiffnessMat = zeros(6, 6)
            LatticeDataPath = MaterialDataDirPath * MaterialFormula * "/" * "Lattice.dat"
            StiffnessDataPath = MaterialDataDirPath * MaterialFormula * "/" * "Stiffness.dat"
            
            if c2dbFlag
                c11, c12, c13, c22, c23, c33 = readdlm(
                    StiffnessDataPath,
                    ',',
                    Float64
                )

                StiffnessMat = [
                    c11 c12 0.0 c13 0.0 0.0;
                    c12 c22 0.0 c23 0.0 0.0;
                    0.0 0.0 0.0 0.0 0.0 0.0;
                    c13 c23 0.0 c33 0.0 0.0;
                    0.0 0.0 0.0 0.0 0.0 0.0;
                    0.0 0.0 0.0 0.0 0.0 0.0
                ]
            else
                lineCount = parse(
                    Int64,
                    read(
                        pipeline(`wc -l $(StiffnessDataPath)`, `awk '{print $1}'`),
                        String
                    )
                )
                if lineCount > 0
                    TempMat = readdlm(
                        LatticeDataPath
                    )
                    hight = TempMat[3, 3]

                    TempMat = readdlm(
                        StiffnessDataPath,
                        skipstart = 3
                    )
                    StiffnessMat .= TempMat[1:end - 1, 2:end]
                    StiffnessMat .*= hight / 100
                else
                    push!(ErrorIndics, index)
                end
            end

            C11Array[index] = StiffnessMat[1, 1]
            C12Array[index] = StiffnessMat[1, 2]
            C13Array[index] = StiffnessMat[1, 4]
            C22Array[index] = StiffnessMat[2, 2]
            C23Array[index] = StiffnessMat[2, 4]
            C33Array[index] = StiffnessMat[4, 4]
        end

        MaterialDF = DataFrame(
            "formula"   =>  MaterialFormulaArray,
            "c_11"      =>  C11Array,
            "c_12"      =>  C12Array,
            "c_13"      =>  C13Array,
            "c_22"      =>  C22Array,
            "c_23"      =>  C23Array,
            "c_33"      =>  C33Array
        )
        MaterialDF = MaterialDF[Not(ErrorIndics), :]

        CSV.write(FixedMaterialDataPath, MaterialDF)
    end

    return MaterialDF, ElementsDict
end


end