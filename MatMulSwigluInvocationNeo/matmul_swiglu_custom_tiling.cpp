/**
 * @file matmul_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;
// typically tiling size wont be greater than 32k
#define TILING_MAX_LEN 32768

//收到tilingData数据结构，将其保存到buf所指向的地址下，并返回首地址buf
uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    if ((tilingSize == 0) || (tilingSize > TILING_MAX_LEN)) {
        assert(false && "Invalid tiling size.");
    }
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

uint8_t *GenerateTiling()
{
    int M = 128;
    int K = 2048;
    int N = 12288; //除以2的原因是B矩阵在n维度上要分为gate矩阵和up矩阵
    // A @ B = C
    // 设置A矩阵
    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;
    // 设置B矩阵
    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;
    // 设置C矩阵
    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT16;
    // 不设置偏置项
    bool isBias = false;

    int usedCoreNum = 32;
    int32_t baseM = 128;
    int32_t baseN = 128;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    // 实例化tilingApi
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K); // 设置MatMul计算时的原始完整的形状
    tilingApi.SetShape(M, N, K); // 设置MatMul单次计算的形状
    tilingApi.SetBias(isBias);
    
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM); // 输出矩阵为M * N, AIcore从M维这边开始遍历
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return GetTilingBuf(&tilingData);
}
