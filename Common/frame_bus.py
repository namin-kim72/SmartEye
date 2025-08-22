from __future__ import annotations
import threading
import time
from typing import Dict, Optional, Tuple, Iterator
import queue
import numpy as np

class FrameBus:
    """
    메인 스레드에서 publish(frame) 호출로 프레임을 배포.
    - 각 모드는 subscribe()로 큐를 받아 프레임 소비. pull(get_latest)도 가능.
    - close() 시 모든 구독자에게 None 브로드캐스트.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._subs: Dict[int, queue.Queue] = {}
        self._next_id = 1
        self._closed = False

    def publish(self, frame: np.ndarray):
        """새로운 프레임을 모든 구독자에게 전송."""
        if self._closed:
            return
        with self._lock:
            self._latest = frame
            self._ts = time.time()
            subs = list(self._subs.values())
        # non-blocking broadcast(오버플로우 시 가장 오래된 항목 삭제하고 최신만 유지)
        for q in subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # 중복된 예외 처리 코드 제거, 하나의 시도로 충분
                q.get_nowait()  # 가장 오래된 항목을 제거
                q.put_nowait(frame)  # 새로운 프레임 삽입

    def get_latest(self, copy: bool = False) -> Tuple[Optional[np.ndarray], float]:
        """최신 프레임을 반환 (복사본 반환 여부 설정 가능)"""
        with self._lock:
            # 복사본을 반환할 때만 np.copy() 사용
            f = None if self._latest is None else (self._latest.copy() if copy else self._latest)
            ts = self._ts
        return f, ts

    def subscribe(self, queue_size: int = 1) -> int:
        """구독자 등록. 작은 큐 크기(1~2)가 지연 누적을 줄임."""
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subs[sub_id] = queue.Queue(maxsize=queue_size)
            return sub_id

    def get_queue(self, sub_id: int) -> queue.Queue:
        """구독자가 사용할 큐 반환"""
        with self._lock:
            return self._subs[sub_id]

    def unsubscribe(self, sub_id: int):
        """구독 해제"""
        with self._lock:
            self._subs.pop(sub_id, None)

    def close(self):
        """모든 구독자에게 종료 시그널 전달"""
        with self._lock:
            self._closed = True
            subs = list(self._subs.values())
            self._subs.clear()
        # 종료 시그널을 보낼 때 예외 처리가 한번만 수행되도록 변경
        for q in subs:
            try:
                q.put_nowait(None)  # 종료 시그널
            except queue.Full:
                pass

    def iter_subscriber(self, sub_id: int, timeout: Optional[float] = 1.0) -> Iterator[Optional[np.ndarray]]:
        """구독자의 프레임을 받아오는 반복자"""
        q = self.get_queue(sub_id)
        while True:
            try:
                item = q.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is None and self._closed:
                break
            yield item
