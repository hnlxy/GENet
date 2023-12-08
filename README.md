## Prerequisites

The code is built with following libraries:
* PyTorch, torchvision
* tensorboardx

## Data Preparation

For GENet, we need to first extract videos into frames for all datasets ([Something-Something V1](https://20bn.com/datasets/something-something/v1), [Something-Something V2](https://20bn.com/datasets/something-something/v2), and [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)), following the [TSN](https://github.com/yjxiong/temporal-segment-networks) repo.


## Code

GENet codes are based on [TSN](https://github.com/yjxiong/temporal-segment-networks) codebases.

## Train 

```shell
sh train.sh
```

## Test

```shell
sh test.sh
```

## Acknowledgement

Thanks for the following Github projects:
- https://github.com/yjxiong/temporal-segment-networks
- https://github.com/mit-han-lab/temporal-shift-module
- https://github.com/haoyanbin918/group-contextualization

