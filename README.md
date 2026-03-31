# Ollama Serving Benchmark Tool

> **ShareGPT 데이터셋을 활용하여 Ollama 모델의 서빙 성능(Throughput, 지연 시간 등)을 측정하는 비동기 파이썬 벤치마크 스크립트**

이 스크립트는 vLLM 벤치마크와 유사한 형태의 성능 지표를 출력하도록 설계되었습니다. Ollama API에 비동기 스트리밍 요청을 보내어 첫 토큰 생성 시간(TTFT), 토큰 당 생성 시간(TPOT), 토큰 간 지연 시간(ITL) 및 전체 처리량(Throughput)을 정밀하게 측정합니다.

---

## 핵심 기능 (Features)
* **비동기 처리 (Asynchronous):** `aiohttp`와 `asyncio`를 활용하여 지정된 동시성(Concurrency) 수준에 맞춰 API 요청을 병렬 처리.
* **ShareGPT 데이터셋 연동:** 대화형 프롬프트 데이터셋을 파싱하여 실제 사용 환경과 유사한 벤치마크 환경 구성.
* **vLLM 호환 지표 출력:**
  * **TTFT (Time to First Token):** 프롬프트 입력 후 첫 토큰이 생성되기까지의 시간
  * **TPOT (Time per Output Token):** 첫 토큰 이후, 출력 토큰 1개당 평균 생성 시간
  * **ITL (Inter-token Latency):** 생성되는 토큰과 토큰 사이의 지연 시간
  * **Throughput:** 초당 요청 처리량(req/s) 및 토큰 처리량(tok/s)
* **JSON 결과 저장:** `--save-results` 플래그를 통해 상세 측정 결과를 JSON 파일로 기록.

---

## 설치 및 준비 (Requirements)

Python 환경에서 필수 패키지들을 설치해야 합니다. 제공되는 `requirements.txt`를 사용하여 의존성을 설치하세요.

```bash
# 패키지 설치
pip install -r requirements.txt
```

**[requirements.txt 구성]**
```text
aiohttp>=3.8.0
numpy>=1.20.0
tqdm>=4.62.0
```

---

## 사용 방법 (Usage)

벤치마크를 실행하기 전에 Ollama 서버가 백그라운드에서 실행 중이어야 하며, 테스트할 모델이 Pull 되어 있어야 합니다.

```bash
# 기본 실행 예시 (프롬프트 100개, 동시성 4)
python benchmark.py --model "llama3:8b" --dataset-path "sharegpt.json" --num-prompts 100 --concurrency 4
```

### 주요 파라미터 (Arguments)
* `--url`: Ollama API 엔드포인트 (기본값: `http://localhost:11434/api/generate`)
* `--model`: 벤치마크를 수행할 모델명 (기본값: `llama3:8b`)
* `--dataset-path`: ShareGPT 데이터셋 JSON 파일 경로 **(필수)**
* `--num-prompts`: 테스트에 사용할 프롬프트 수 (기본값: 500)
* `--max-tokens`: 생성할 최대 토큰 수 (기본값: 1024)
* `--concurrency`: 동시 요청 수 (기본값: 1)
* `--seed`: 난수 생성 시드 (기본값: 42)
* `--save-results`: 측정 결과를 JSON 파일로 저장

---

## 출력 예시 (Output Example)

실행이 완료되면 터미널에 다음과 같은 형태의 상세 지표가 출력됩니다. (값은 예시입니다)

```text
========== Serving Benchmark Result ==========
Successful requests:                100
Benchmark duration (s):             45.21
Total input tokens:                 4520
Total generated tokens:             21045
Request throughput (req/s):         2.21
Output token throughput (tok/s):    465.50
Total Token throughput (tok/s):     565.47
------------Time to First Token-------------
Mean TTFT (ms):                     124.50000
Median TTFT (ms):                   110.20000
P99 TTFT (ms):                      350.10000
----Time per Output Token (excl. 1st token)-----
Mean TPOT (ms):                     18.50000
Median TPOT (ms):                   17.20000
P99 TPOT (ms):                      45.10000
-------------Inter-token Latency--------------
Mean ITL (ms):                      18.45000
Median ITL (ms):                    17.15000
P99 ITL (ms):                       45.00000
==============================================
```
