# 모델의 입/출력
input_ids - 인코더 입력
attention_mask - 인코더 패딩 마스크
decoder_input_ids - 디코더 입력
decoder_attention_mask - 디코더 패딩 마스크

```
{
    'input_ids': (B, N),
    'attention_mask': (B,N)
    'decoder_input_ids': (B, N)
    'decoder_attention_mask': (B, N)
}
```

# 데이터셋
## KHS
* 레이블: 혐오 / 공격적 / 일반적
* padding: 128

## kornli
레이블: 모순 / 수반 / 관계없음

## nsmc
레이블: 긍정적 / 부정적