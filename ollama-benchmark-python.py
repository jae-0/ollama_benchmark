#!/usr/bin/env python3
"""
Ollama 벤치마크 스크립트 (Python) - vLLM 벤치마크와 호환되는 형식으로 결과 출력
"""

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import numpy as np
from tqdm import tqdm

@dataclass
class SampleRequest:
    """벤치마크용 요청 샘플"""
    prompt: str
    prompt_len: int  # 토큰 수 (추정)

@dataclass
class RequestOutput:
    """요청에 대한 응답 결과"""
    success: bool
    prompt_len: int
    generated_text: str = ""
    output_tokens: int = 0
    ttft: float = 0.0  # 첫 토큰까지의 시간 (초)
    latency: float = 0.0  # 총 지연 시간 (초)
    itl: List[float] = field(default_factory=list)  # 토큰 간 지연 시간 (초)
    error: str = ""

async def load_sharegpt_dataset(dataset_path: str, num_requests: int) -> List[SampleRequest]:
    """ShareGPT 데이터셋에서 프롬프트 샘플링"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for conv in data:
        if 'conversations' in conv:
            for msg in conv.get('conversations', []):
                if msg.get('from') == 'human':
                    # 간단한 토큰 수 추정 (4자당 1토큰으로 가정)
                    prompt_len = len(msg.get('value', '')) // 4
                    prompts.append(SampleRequest(
                        prompt=msg.get('value', '').strip(),
                        prompt_len=prompt_len
                    ))
                    break  # 각 대화에서 첫 번째 인간 메시지만 사용
    
    # 요청 수만큼 샘플링
    if len(prompts) > num_requests:
        prompts = random.sample(prompts, num_requests)
    else:
        # 충분한 프롬프트가 없으면 반복
        while len(prompts) < num_requests:
            prompts.extend(prompts[:num_requests - len(prompts)])
    
    return prompts[:num_requests]

async def ollama_streaming_request(
    session: aiohttp.ClientSession, 
    url: str, 
    model: str, 
    prompt: str, 
    max_tokens: int
) -> Tuple[RequestOutput, List[Dict]]:
    """Ollama API에 스트리밍 요청 전송"""
    start_time = time.time()
    
    # 요청 데이터
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens
        },
        "raw": True  # 로우 모드 활성화 (메타데이터 포함)
    }
    
    try:
        async with session.post(url, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return RequestOutput(
                    success=False,
                    prompt_len=len(prompt) // 4,
                    error=f"HTTP 오류: {resp.status} - {error_text}"
                ), []
            
            # 응답 처리 변수
            streaming_outputs = []
            output_text = ""
            ttft = None
            token_times = []
            
            # 스트리밍 응답 처리
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    current_time = time.time()
                    response = json.loads(line)
                    streaming_outputs.append(response)
                    
                    # 첫 토큰 시간 기록
                    if ttft is None and "response" in response:
                        ttft = current_time - start_time
                        token_times.append(current_time)
                    
                    if "response" in response:
                        output_text += response["response"]
                        if ttft is not None:  # 첫 토큰 이후의 토큰만 시간 기록
                            token_times.append(current_time)
                except json.JSONDecodeError:
                    continue
            
            end_time = time.time()
            
            # 첫 토큰 시간이 기록되지 않은 경우
            if ttft is None:
                return RequestOutput(
                    success=False,
                    prompt_len=len(prompt) // 4,
                    error="첫 토큰 생성 실패"
                ), streaming_outputs
            
            # 토큰 간 지연 시간 계산
            itl = []
            for i in range(1, len(token_times)):
                itl.append(token_times[i] - token_times[i-1])
            
            return RequestOutput(
                success=True,
                prompt_len=len(prompt) // 4,
                generated_text=output_text,
                output_tokens=len(output_text) // 4,  # 추정
                ttft=ttft,
                latency=end_time - start_time,
                itl=itl
            ), streaming_outputs
            
    except Exception as e:
        return RequestOutput(
            success=False,
            prompt_len=len(prompt) // 4,
            error=str(e)
        ), []

async def process_request(session: aiohttp.ClientSession, url: str, model: str, 
                         request: SampleRequest, max_tokens: int, idx: int) -> RequestOutput:
    """단일 요청 처리"""
    print(f"요청 {idx+1} 시작...")
    
    output, _ = await ollama_streaming_request(
        session=session,
        url=url,
        model=model,
        prompt=request.prompt,
        max_tokens=max_tokens
    )
    
    if output.success:
        print(f"요청 {idx+1} 완료: 입력 토큰={output.prompt_len}, 출력 토큰={output.output_tokens}, TTFT={output.ttft*1000:.6f}ms")
    else:
        print(f"요청 {idx+1} 실패: {output.error}")
    
    return output

async def run_benchmark(
    url: str,
    model: str,
    requests: List[SampleRequest],
    max_tokens: int,
    concurrency: int
) -> Tuple[List[RequestOutput], float]:
    """벤치마크 실행"""
    # 벤치마크 시작 시간
    start_time = time.time()
    
    # 세마포어로 동시성 제한
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_process_request(session, url, model, request, max_tokens, idx):
        async with semaphore:
            return await process_request(session, url, model, request, max_tokens, idx)
    
    # 모든 요청 비동기 처리
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(
                bounded_process_request(session, url, model, request, max_tokens, i)
            )
            tasks.append(task)
        
        outputs = await asyncio.gather(*tasks)
    
    # 벤치마크 종료 시간
    end_time = time.time()
    benchmark_duration = end_time - start_time
    
    return outputs, benchmark_duration

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """리스트의 통계 계산 (평균, 중앙값, 표준편차, P99)"""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p99": 0.0
        }
    
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p99": float(np.percentile(values, 99))
    }

def calculate_metrics(outputs: List[RequestOutput], duration: float) -> Dict[str, Any]:
    """벤치마크 결과에서 지표 계산"""
    # 성공한 요청만 필터링
    successful_outputs = [o for o in outputs if o.success]
    
    # 기본 지표
    total_requests = len(outputs)
    completed = len(successful_outputs)
    total_input = sum(o.prompt_len for o in successful_outputs)
    total_output = sum(o.output_tokens for o in successful_outputs)
    
    # 처리량 지표
    request_throughput = completed / duration
    output_throughput = total_output / duration
    total_token_throughput = (total_input + total_output) / duration
    
    # TTFT (Time to First Token)
    ttfts = [o.ttft * 1000 for o in successful_outputs]  # 밀리초 단위
    ttft_stats = calculate_stats(ttfts)
    
    # TPOT (Time per Output Token, excluding first token)
    tpots = []
    for o in successful_outputs:
        if o.output_tokens > 1:
            time_after_first = o.latency - o.ttft
            tpot = time_after_first / (o.output_tokens - 1) * 1000  # 밀리초 단위
            tpots.append(tpot)
    tpot_stats = calculate_stats(tpots)
    
    # ITL (Inter-token Latency)
    all_itls = []
    for o in successful_outputs:
        all_itls.extend([itl * 1000 for itl in o.itl])  # 밀리초 단위
    itl_stats = calculate_stats(all_itls)
    
    return {
        "total_requests": total_requests,
        "completed": completed,
        "failed": total_requests - completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": duration,
        "request_throughput": request_throughput,
        "output_throughput": output_throughput,
        "total_token_throughput": total_token_throughput,
        "ttft": ttft_stats,
        "tpot": tpot_stats,
        "itl": itl_stats
    }

def print_benchmark_results(metrics: Dict[str, Any]):
    """벤치마크 결과 출력"""
    print("========== Serving Benchmark Result ==========")
    print(f"Successful requests:                {metrics['completed']}")
    print(f"Benchmark duration (s):             {metrics['duration']:.2f}")
    print(f"Total input tokens:                 {metrics['total_input']}")
    print(f"Total generated tokens:             {metrics['total_output']}")
    print(f"Request throughput (req/s):         {metrics['request_throughput']:.2f}")
    print(f"Output token throughput (tok/s):    {metrics['output_throughput']:.2f}")
    print(f"Total Token throughput (tok/s):     {metrics['total_token_throughput']:.2f}")
    
    print("------------Time to First Token-------------")
    print(f"Mean TTFT (ms):                     {metrics['ttft']['mean']:.5f}")
    print(f"Median TTFT (ms):                   {metrics['ttft']['median']:.5f}")
    print(f"P99 TTFT (ms):                      {metrics['ttft']['p99']:.5f}")
    
    print("----Time per Output Token (excl. 1st token)-----")
    print(f"Mean TPOT (ms):                     {metrics['tpot']['mean']:.5f}")
    print(f"Median TPOT (ms):                   {metrics['tpot']['median']:.5f}")
    print(f"P99 TPOT (ms):                      {metrics['tpot']['p99']:.5f}")
    
    print("-------------Inter-token Latency--------------")
    print(f"Mean ITL (ms):                      {metrics['itl']['mean']:.5f}")
    print(f"Median ITL (ms):                    {metrics['itl']['median']:.5f}")
    print(f"P99 ITL (ms):                       {metrics['itl']['p99']:.5f}")
    
    print("==============================================")

def save_results(metrics: Dict[str, Any], args: argparse.Namespace, outputs: List[RequestOutput]):
    """결과를 JSON 파일로 저장"""
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 결과 딕셔너리 생성
    results = {
        "date": timestamp,
        "model": args.model,
        "dataset": args.dataset_path,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        
        # 벤치마크 성능 지표
        "duration": metrics["duration"],
        "completed": metrics["completed"],
        "failed": metrics["failed"],
        "total_input_tokens": metrics["total_input"],
        "total_output_tokens": metrics["total_output"],
        "request_throughput": metrics["request_throughput"],
        "output_throughput": metrics["output_throughput"],
        "total_token_throughput": metrics["total_token_throughput"],
        
        # 상세 지표
        "mean_ttft_ms": metrics["ttft"]["mean"],
        "median_ttft_ms": metrics["ttft"]["median"],
        "p99_ttft_ms": metrics["ttft"]["p99"],
        
        "mean_tpot_ms": metrics["tpot"]["mean"],
        "median_tpot_ms": metrics["tpot"]["median"],
        "p99_tpot_ms": metrics["tpot"]["p99"],
        
        "mean_itl_ms": metrics["itl"]["mean"],
        "median_itl_ms": metrics["itl"]["median"],
        "p99_itl_ms": metrics["itl"]["p99"],
    }
    
    # 파일명 생성
    model_id = args.model.replace("/", "-").replace(":", "-")
    filename = f"ollama-benchmark-{model_id}-{timestamp}.json"
    
    # 결과 저장
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"결과가 {filename}에 저장되었습니다.")

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Ollama 벤치마크 스크립트 (Python)")
    parser.add_argument("--url", type=str, default="http://localhost:11434/api/generate",
                      help="Ollama API URL")
    parser.add_argument("--model", type=str, default="llama3:8b",
                      help="벤치마크할 모델명")
    parser.add_argument("--dataset-path", type=str, required=True,
                      help="ShareGPT 데이터셋 경로")
    parser.add_argument("--num-prompts", type=int, default=500,
                      help="처리할 프롬프트 수 (기본값: 500)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                      help="생성할 최대 토큰 수 (기본값: 1024)")
    parser.add_argument("--concurrency", type=int, default=1,
                      help="동시 요청 수 (기본값: 1)")
    parser.add_argument("--seed", type=int, default=42,
                      help="난수 시드 (기본값: 42)")
    parser.add_argument("--save-results", action="store_true",
                      help="결과를 JSON 파일로 저장")
    
    args = parser.parse_args()
    
    # 난수 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("========== ShareGPT Ollama 벤치마크 ==========")
    print(f"모델: {args.model}")
    print(f"동시성: {args.concurrency}")
    print(f"요청 수: {args.num_prompts}")
    print(f"최대 토큰: {args.max_tokens}")
    print(f"데이터셋: {args.dataset_path}")
    print("==============================================")
    
    # 데이터셋 로드
    print("데이터셋에서 프롬프트 추출 중...")
    requests = await load_sharegpt_dataset(args.dataset_path, args.num_prompts)
    print(f"추출된 프롬프트 수: {len(requests)}")
    
    # 벤치마크 실행
    outputs, duration = await run_benchmark(
        url=args.url,
        model=args.model,
        requests=requests,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency
    )
    
    # 지표 계산
    metrics = calculate_metrics(outputs, duration)
    
    # 결과 출력
    print_benchmark_results(metrics)
    
    # 결과 저장 (선택사항)
    if args.save_results:
        save_results(metrics, args, outputs)
    
    print("벤치마크 완료!")

if __name__ == "__main__":
    asyncio.run(main())
