# 모듈
훈련:
```
datamodule
  - datapipe
  - tokenizer
  - feature converter

lightning model
  - tokenizer
  - model
```

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

# 모델 별 노트
## KeT5의 입출력
- EOS 토큰: `</s>`
- BOS 토큰: 없음, 대신 padding token을 사용함. (`model.config.decoder_start_token_id`에서 알 수 있음)
**Implementation Detail**: [seqio source](https://github.com/google/seqio/blob/main/seqio/feature_converters.py#L259) 에서 실제로 bos_id로 pad_token의 id를 사용한다는 걸 알 수 있음

- 인코더 입력: 안녕`</s>`
- 디코더 입력: `[PAD]`Hello
- 디코더 정답: Hello



# 데이터셋
## KHS
* 레이블: 혐오 / 공격적 / 일반적
* padding: 128

## kornli
레이블: 모순 / 수반 / 관계없음

## nsmc
레이블: 긍정적 / 부정적