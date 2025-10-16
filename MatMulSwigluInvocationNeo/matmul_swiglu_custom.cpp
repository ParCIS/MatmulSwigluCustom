/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;


constexpr int32_t BUFFER_NUM = 2; // 双缓冲

//实现将GM上的二进制Tiling结构体数据恢复成本地的C++结构体
__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    // printf("M: %d\tsingleCoreM: %d\tbaseM: %d\n", tiling->M, tiling->singleCoreM, tiling->baseM);
    // printf("N: %d\tsingleCoreN: %d\tbaseN: %d\n", tiling->N, tiling->singleCoreN, tiling->baseN);
    // printf("K: %d\tsingleCoreK: %d\tbaseK: %d\n", tiling->Ka, tiling->singleCoreK, tiling->baseK);
    return;
}

template <typename aType, typename bType, typename cType> class MatmulSwigluKernel {
public:
    __aicore__ inline MatmulSwigluKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling, TPipe *pipe);
    __aicore__ inline void Process();

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void SwigluCompute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC);
    
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, aType>, MatmulType<TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, cType>> matmulGateObj, matmulUpObj; //分别进行gate和up矩阵运算
    GlobalTensor<aType> aGlobal;
    GlobalTensor<aType> bGlobal;
    GlobalTensor<bType> bGateGlobal, bUpGlobal; //分别存放gateGM首地址和UpGM首地址
    GlobalTensor<cType> cGlobal;
    GlobalTensor<cType> workspaceGlobal; // 用于Matmul计算时的额外空间
    GlobalTensor<cType> gateworkspaceGlobal, upworkspaceGlobal; // 用于Matmul计算时的额外空间
    LocalTensor<cType> gateInLocal, upInLocal; // 作为UB中的一块缓冲区，用于接受matmul的计算结果
    TCubeTiling tiling;
    TQue<QuePosition::VECIN, BUFFER_NUM> gateInQueue, upInQueue; //分别用于存放gate和up矩阵做完矩阵乘后的数据块
    TQue<QuePosition::VECOUT, BUFFER_NUM> swigluOutQueue; // 用于存放swiglu计算完成，等待拷贝回GM的数据块
    TBuf<QuePosition::VECCALC> calcBufs; // 用于swiglu计算的临时变量
    uint32_t splitRowNums = 2; //固定的切分数，计算出每个更小行快的大小，因为向量单元一次能处理的数据量有限(256B)，切分可以更好匹配硬件能力
    uint32_t splitRowSize = 0;
    bool isTransA = false;
    bool isTransB = true;
};


template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulSwigluKernel<aType, bType, cType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, 
                                                                    const TCubeTiling &tiling, TPipe *pipe)
{
    this->tiling = tiling;
    splitRowSize = tiling.baseM / splitRowNums; 
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N * 2); //里面包含gate和up
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(workspace), tiling.M * tiling.N * 2);
    int32_t offsetA, offsetB, offsetC;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC);
    aGlobal = aGlobal[offsetA];
    bGateGlobal = bGlobal[offsetB];
    bUpGlobal = bGlobal[tiling.N + offsetB];
    if (isTransB) {
        bUpGlobal = bGlobal[tiling.Kb * tiling.N + offsetB];
    }
    cGlobal = cGlobal[offsetC];
    gateworkspaceGlobal = workspaceGlobal[GetBlockIdx() * tiling.singleCoreM * tiling.singleCoreN * 2];
    upworkspaceGlobal = workspaceGlobal[GetBlockIdx() * tiling.singleCoreM * tiling.singleCoreN * 2 + tiling.singleCoreM * tiling.singleCoreN];
    //用于Matmul --> swiglu 的过程中的同步队列
    pipe->InitBuffer(gateInQueue, BUFFER_NUM, tiling.baseM * tiling.baseN * sizeof(cType));
    pipe->InitBuffer(upInQueue, BUFFER_NUM, tiling.baseM * tiling.baseN * sizeof(cType));
    //用于swiglu --> GM 的过程中的同步队列
    pipe->InitBuffer(swigluOutQueue, BUFFER_NUM, splitRowSize * tiling.baseN * sizeof(cType));
    pipe->InitBuffer(calcBufs, splitRowSize * tiling.baseN * 2);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulSwigluKernel<aType, bType, cType>::Process()
{
    matmulGateObj.SetWorkspace(gateworkspaceGlobal);
    matmulGateObj.SetTensorA(aGlobal, isTransA);
    matmulGateObj.SetTensorB(bGateGlobal, isTransB);
    matmulGateObj.template Iterate<false>();
    matmulUpObj.SetWorkspace(upworkspaceGlobal);
    matmulUpObj.SetTensorA(aGlobal, isTransA);
    matmulUpObj.SetTensorB(bUpGlobal, isTransB);
    matmulUpObj.template Iterate<false>();
    //遍历当前AIcore被分配的大块中的所有小块
    for (int i = 0; i < tiling.singleCoreM * tiling.singleCoreN / (tiling.baseM * tiling.baseN); ++i) {
        MatmulCompute();
        gateInLocal = gateInQueue.DeQue<cType>();
        upInLocal = upInQueue.DeQue<cType>();
        //当前小块又被分成splitRowNums个小小块
        for (int j = 0; j < splitRowNums; ++j) {
            SwigluCompute(j);
            CopyOut(i * splitRowNums + j); // i * splitRowNums + j 是偏移值
        }
        gateInQueue.FreeTensor(gateInLocal);
        upInQueue.FreeTensor(upInLocal);
    }
    matmulGateObj.End();
    matmulUpObj.End();
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulSwigluKernel<aType, bType, cType>::MatmulCompute()
{
    gateInLocal = gateInQueue.AllocTensor<cType>();
    matmulGateObj.template GetTensorC<false>(gateInLocal, false, true);
    gateInQueue.EnQue(gateInLocal);
    upInLocal = upInQueue.AllocTensor<cType>();
    matmulUpObj.template GetTensorC<false>(upInLocal, false, true);
    upInQueue.EnQue(upInLocal);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulSwigluKernel<aType, bType, cType>::SwigluCompute(uint32_t count)
{
    auto swigluOutLocal = swigluOutQueue.AllocTensor<cType>();
    auto tmpLocal = calcBufs.Get<uint8_t>();
    float alpha = 1;
    SwiGLU<cType, false>(swigluOutLocal, upInLocal[count * splitRowSize * tiling.baseN], 
                         gateInLocal[count * splitRowSize * tiling.baseN], alpha, tmpLocal, splitRowSize * tiling.baseN);
    swigluOutQueue.EnQue(swigluOutLocal);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulSwigluKernel<aType, bType, cType>::CopyOut(uint32_t count)
{
    auto swigluOutLocal = swigluOutQueue.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / splitRowSize;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * splitRowSize * tiling.N + count / roundM * tiling.baseN);
    DataCopyParams copyParam = {(uint16_t)splitRowSize, (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE), 0,
                                (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], swigluOutLocal, copyParam);
    swigluOutQueue.FreeTensor(swigluOutLocal);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulSwigluKernel<aType, bType, cType>::CalcOffset(int blockIdx, const TCubeTiling &tiling, 
                                                                           int &offsetA, int &offsetB, int &offsetC)
{
    uint32_t mSingleBlocks = Ceil(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    if (isTransA) {
        offsetA = mCoreIndx * tiling.singleCoreM;
    }
    offsetB = nCoreIndx * tiling.singleCoreN;
    if (isTransB) {
        offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
    }
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_swiglu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    MatmulSwigluKernel<half, half, half> matmulswigluKernel;
    matmulswigluKernel.Init(a, b, c, workspace, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulswigluKernel.matmulUpObj, &matmulswigluKernel.tiling, matmulswigluKernel.matmulGateObj, &matmulswigluKernel.tiling);
    matmulswigluKernel.Process();
}

