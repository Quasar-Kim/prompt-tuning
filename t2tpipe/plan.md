# Refactoring
## done
- typing 완전 재정비 -> type: ignore 줄이기
  * Protocol 사용시 오류 많이 발생 -> 클래스는 모두 ABC로 바꾸기, 단 dict는 그대로 protocol 사용
  * typing 분리
  * Sample -> TextSample -> EncodedSample -> Batch -> logits(tensor) -> TextPrediction -> Prediction
- 클래스 그 자체를 넘겨주는 패턴 피하기
  * 현재는 Model() definition에서 사용하고 있음
  * __init__()은 정적 configuration에만 사용
  * setup(env) 메소드 도입하기
    + env.model -> Model()
      env.task -> Task()
      env.datamodule -> DataModule()
      env.runtime_config -> assemble()의 runtime_config dict
      (확장성 때문에 kwargs 사용 x)
- runtime config와 다른 config 확실히 분리
  * pad_to: 지금은 create_experiment kwargs -> Padder().__init__() kwargs로
  * Task 정의시 config 넘겨줄 수 있도록: Task(datamodule_config: dict)

- default 다루는 패턴 하나로 정하기
  * kwargs: literal(int, str...)인 경우 default 명시, instance인 경우 None + if statement
  * dict: { 'key': value, **config }
- 함수 signature를 확장하는 경우 상위 함수에서는 *args, **kwargs로, 같은 parameter 두번 정의 피하기
  NOTE: signautre 한번 바꾸려고 하면 너무 어려움.
        그리고 intellisense 안된다고 안하는 것은 말도 안되는 결정 - 유지보수의 용이성 > intellisense 용이성
        typing이 안되는 문제는 어쩔 수 없음 - python은 inherently dynamic language이므로 burden of static typing-based solution > simplicity of dynamic solution 시 당연히 dynamic solution을 선택하는 것이 맞다.
- 이름 변경
  * Model(module) -> Model(lightning_module)
  * create_experiment(**kwargs) -> assemble(runtime_config: dict) (확장성 때문에 kwargs 사용 x)
  * 내부 오브젝트 중 DataPipe로 시작하는 것들 T2TPipe로 시작하도록
  * 이름 확실히 구분: T2TPipe - 이 패키지, TorchDataPipe - pytorch datapipe, DataPipe - 파이프 객체 (네이티브 객체 pipe와 이름 충돌 회피)
- core.py에서 직접 import를 피하고, __init__ 사용
  NOTE: 필요하면 직접 import하면 되므로 public interface 딱 3개만 export하고 건드리지 말기
- linter: 그냥 black 써
- pipe 명시적으로
  * DistributedShardingFilter 도입 - dataset에서 distributed sharding 기능 offload
  * ShardingFilter 도입 - WorkerShardingFilter + DistributedShardingFilter
    NOTE: 현재 implementation은 shuffling이 제대로 되지 않는 문제 발생

## TODO
- pipe 명시적으로
  * Pipe 중간에 들어거야 할 것은 <name>Placeholder()
    ex) TemplatePlaceHolder()

# Feature
- GPTFeatureConverter
- TemplatingMapper
