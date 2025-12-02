import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI, Type } from "@google/genai";

// --- Constants & Types ---

const GRID_W = 60;
const GRID_H = 30;
const CELL_SIZE = 24;
const BASE_TICK_RATE_MS = 100;
const MOVEMENT_SPEED = 0.15;
const DEADLOCK_THRESHOLD_TICKS = 300;
const RESERVATION_LOOKAHEAD = 6;

type CellType = 'empty' | 'track' | 'station';

interface Cell {
  x: number;
  y: number;
  type: CellType;
  stationName?: string;
  id: string;
}

interface Train {
  id: string;
  name: string;
  fromId: string;
  toId: string;
  path: string[];
  currentPathIndex: number;
  progress: number;
  state: 'moving' | 'waiting' | 'arrived' | 'dwelling';
  priority: number;
  type: 'manual' | 'schedule' | 'batch';
  color: string;
  waitingTime: number;
  isCyclic: boolean;
  dwellRemaining: number;
}

interface ScheduleItem {
  id: string;
  time: number;
  from: string;
  to: string;
  priority: number;
  dispatched: boolean;
}

interface HistoryItem {
  id: string;
  time: number;
  from: string;
  to: string;
  priority: number;
  type: 'manual' | 'schedule' | 'batch';
  isCyclic: boolean;
}

// --- Helper Functions ---

const getCellId = (x: number, y: number) => `${x}-${y}`;
const parseCellId = (id: string | undefined) => {
  if (!id) return { x: 0, y: 0 };
  const [x, y] = id.split('-').map(Number);
  return { x, y };
};

// Expanded color palette
const TRAIN_COLORS = [
  '#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981',
  '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef', '#f43f5e',
  '#ec4899', '#14b8a6', '#6366f1', '#a855f7', '#fbbf24',
  '#a3e635', '#22d3ee', '#818cf8', '#c084fc', '#fb7185',
  '#34d399', '#fcd34d', '#fb923c', '#4ade80', '#2dd4bf'
];

// --- 优化的二叉堆优先队列 ---
class MinHeap<T extends { f: number }> {
  private heap: T[] = [];

  private parent(i: number): number { return Math.floor((i - 1) / 2); }
  private leftChild(i: number): number { return 2 * i + 1; }
  private rightChild(i: number): number { return 2 * i + 2; }

  private swap(i: number, j: number): void {
    [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
  }

  private siftUp(i: number): void {
    while (i > 0 && this.heap[this.parent(i)].f > this.heap[i].f) {
      this.swap(this.parent(i), i);
      i = this.parent(i);
    }
  }

  private siftDown(i: number): void {
    let minIndex = i;
    const left = this.leftChild(i);
    const right = this.rightChild(i);

    if (left < this.heap.length && this.heap[left].f < this.heap[minIndex].f) {
      minIndex = left;
    }
    if (right < this.heap.length && this.heap[right].f < this.heap[minIndex].f) {
      minIndex = right;
    }

    if (i !== minIndex) {
      this.swap(i, minIndex);
      this.siftDown(minIndex);
    }
  }

  push(item: T): void {
    this.heap.push(item);
    this.siftUp(this.heap.length - 1);
  }

  pop(): T | undefined {
    if (this.heap.length === 0) return undefined;
    if (this.heap.length === 1) return this.heap.pop();
    
    const result = this.heap[0];
    this.heap[0] = this.heap.pop()!;
    this.siftDown(0);
    return result;
  }

  isEmpty(): boolean {
    return this.heap.length === 0;
  }

  size(): number {
    return this.heap.length;
  }
}

// --- 旧版优先队列 (用于性能对比) ---
class PriorityQueueLegacy<T extends { f: number }> {
  private items: T[] = [];
  push(item: T) {
    this.items.push(item);
    this.items.sort((a, b) => a.f - b.f); // 性能瓶颈：每次插入都排序 O(N log N)
  }
  pop(): T | undefined {
    return this.items.shift();
  }
  isEmpty() {
    return this.items.length === 0;
  }
}

// --- A* Node ---
interface AStarNode {
  id: string;
  x: number;
  y: number;
  g: number;
  h: number;
  f: number;
  parent?: AStarNode;
}

const getHeuristic = (x1: number, y1: number, x2: number, y2: number): number => {
  // 曼哈顿距离 + 微小的对角线惩罚
  return Math.abs(x1 - x2) + Math.abs(y1 - y2);
};

// --- Component ---

function App() {
  // --- State ---
  const [grid, setGrid] = useState<Record<string, Cell>>({});
  const [editMode, setEditMode] = useState<'track' | 'station'>('track');
  
  const [trains, setTrains] = useState<Train[]>([]);
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [schedule, setSchedule] = useState<ScheduleItem[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  
  const [aiPrompt, setAiPrompt] = useState("");
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [apiKeyMissing, setApiKeyMissing] = useState(false);

  const [dispatchFrom, setDispatchFrom] = useState("");
  const [dispatchTo, setDispatchTo] = useState("");
  const [dispatchPriority, setDispatchPriority] = useState(5);
  const [dispatchIsCyclic, setDispatchIsCyclic] = useState(false);

  const [historyFilterFrom, setHistoryFilterFrom] = useState("");
  const [historyFilterTo, setHistoryFilterTo] = useState("");

  const [hoveredTrainId, setHoveredTrainId] = useState<string | null>(null);
  const [previewPath, setPreviewPath] = useState<string[]>([]);
  const [globalDeadlockWarning, setGlobalDeadlockWarning] = useState(false);
  
  // 新增：速度控制
  const [simulationSpeed, setSimulationSpeed] = useState(1);
  
  // 新增：批量发车设置
  const [batchCount, setBatchCount] = useState(10);
  const [batchIsCyclic, setBatchIsCyclic] = useState(true);
  const [batchDelay, setBatchDelay] = useState(5); // 每隔多少tick发一辆

  // 性能测试结果状态
  const [benchmarkResult, setBenchmarkResult] = useState<{
    legacyTime: number;
    newTime: number;
    iterations: number;
    improvement: string;
  } | null>(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  // Refs
  const gridRef = useRef(grid);
  const trainsRef = useRef(trains);
  const scheduleRef = useRef(schedule);
  const timeRef = useRef(time);
  
  gridRef.current = grid;
  trainsRef.current = trains;
  scheduleRef.current = schedule;
  timeRef.current = time;

  const isMouseDown = useRef(false);
  const dragAction = useRef<'track' | 'erase' | null>(null);
  
  // 批量发车队列
  const batchQueueRef = useRef<{from: string, to: string, dispatchAt: number, priority: number, isCyclic: boolean}[]>([]);

  // Initialize Grid
  useEffect(() => {
    const initialGrid: Record<string, Cell> = {};
    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        const id = getCellId(x, y);
        initialGrid[id] = { x, y, type: 'track', id };
      }
    }
    const addStation = (x: number, y: number, name: string) => {
      const id = getCellId(x, y);
      initialGrid[id] = { ...initialGrid[id], type: 'station', stationName: name };
    };
    
    // 默认车站 - 适应大地图
    addStation(5, 15, "西站");
    addStation(55, 15, "东站");
    addStation(30, 5, "北站");
    addStation(30, 25, "南站");
    
    addStation(15, 10, "西北站");
    addStation(45, 10, "东北站");
    addStation(15, 20, "西南站");
    addStation(45, 20, "东南站");

    setGrid(initialGrid);
  }, []);

  useEffect(() => {
    if (!process.env.API_KEY) {
      setApiKeyMissing(true);
    }
  }, []);

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      isMouseDown.current = false;
      dragAction.current = null;
    };
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // --- Helpers ---

  const getUniqueColor = useCallback((currentTrains: Train[]) => {
    const usedColors = new Set(currentTrains.map(t => t.color));
    const available = TRAIN_COLORS.filter(c => !usedColors.has(c));
    if (available.length > 0) {
      return available[Math.floor(Math.random() * available.length)];
    }
    return TRAIN_COLORS[Math.floor(Math.random() * TRAIN_COLORS.length)];
  }, []);

  const resetSystem = useCallback(() => {
    setIsRunning(false);
    setTime(0);
    setTrains([]);
    setSchedule([]);
    setHistory([]);
    setGlobalDeadlockWarning(false);
    setDispatchFrom("");
    setDispatchTo("");
    setDispatchIsCyclic(false);
    batchQueueRef.current = [];
  }, []);

  // 清除所有运行中的车辆
  const clearAllTrains = useCallback(() => {
    setTrains([]);
    setGlobalDeadlockWarning(false);
    batchQueueRef.current = [];
  }, []);

  // 一键清除轨道
  const clearTracks = useCallback(() => {
    setGrid(prev => {
      const next = { ...prev };
      Object.values(next).forEach(cell => {
        if (cell.type === 'track') {
          next[cell.id] = { ...cell, type: 'empty' };
        }
      });
      return next;
    });
  }, []);

  // --- 优化的 A* 路径搜索 ---
  const findPathAStar = useCallback((
    startId: string, 
    endId: string, 
    currentGrid: Record<string, Cell>,
    currentTrains: Train[] = [], 
    ignoreTrainId?: string,
    reservedCells?: Set<string>,
    strictMode: boolean = false
  ): string[] | null => {
    
    if (!currentGrid[startId] || !currentGrid[endId]) return null;

    const occupiedCells = new Set<string>();
    const nearCongestionCells = new Set<string>();

    currentTrains.forEach(t => {
      if (t.id === ignoreTrainId) return;
      const tPos = t.path[t.currentPathIndex];
      if (!tPos) return;

      occupiedCells.add(tPos);
      
      const pos = parseCellId(tPos);
      const neighbors = [[0,1], [0,-1], [1,0], [-1,0]];
      neighbors.forEach(([dx, dy]) => {
        nearCongestionCells.add(getCellId(pos.x + dx, pos.y + dy));
      });
    });

    const startNode = parseCellId(startId);
    const endNode = parseCellId(endId);

    // 使用优化的二叉堆
    const openSet = new MinHeap<AStarNode>();
    const startH = getHeuristic(startNode.x, startNode.y, endNode.x, endNode.y);
    openSet.push({ 
      id: startId, 
      x: startNode.x, 
      y: startNode.y, 
      g: 0, 
      h: startH, 
      f: startH 
    });

    const visited = new Map<string, number>();
    visited.set(startId, 0);

    while (!openSet.isEmpty()) {
      const current = openSet.pop()!;

      if (current.id === endId) {
        const path: string[] = [];
        let curr: AStarNode | undefined = current;
        while (curr) {
          path.unshift(curr.id);
          curr = curr.parent;
        }
        return path;
      }

      // 如果当前节点的g值比已记录的大,跳过(已有更优路径)
      if (visited.has(current.id) && visited.get(current.id)! < current.g) {
        continue;
      }

      const neighbors = [
        { nx: current.x, ny: current.y - 1 },
        { nx: current.x, ny: current.y + 1 },
        { nx: current.x - 1, ny: current.y },
        { nx: current.x + 1, ny: current.y }
      ];

      for (const { nx, ny } of neighbors) {
        const nid = getCellId(nx, ny);
        const neighborCell = currentGrid[nid];

        if (neighborCell && (neighborCell.type === 'track' || neighborCell.type === 'station')) {
          let moveCost = 1;
          
          if (strictMode && reservedCells && reservedCells.has(nid) && nid !== endId) {
            continue;
          }

          // 动态权重策略
          if (occupiedCells.has(nid) && nid !== endId) {
            moveCost += 20; 
          }

          if (!strictMode && reservedCells && reservedCells.has(nid) && nid !== endId) {
            moveCost += 1000;
          }

          if (nearCongestionCells.has(nid)) {
            moveCost += 2;
          }

          // 转向惩罚
          if (current.parent) {
            const dx = nx - current.x;
            const dy = ny - current.y;
            const pdx = current.x - current.parent.x;
            const pdy = current.y - current.parent.y;
            if (dx !== pdx || dy !== pdy) {
              moveCost += 0.5;
            }
          }

          const newG = current.g + moveCost;
          
          if (!visited.has(nid) || newG < visited.get(nid)!) {
            visited.set(nid, newG);
            const h = getHeuristic(nx, ny, endNode.x, endNode.y);
            
            const neighborNode: AStarNode = {
              id: nid,
              x: nx,
              y: ny,
              g: newG,
              h: h,
              f: newG + h,
              parent: current
            };
            openSet.push(neighborNode);
          }
        }
      }
    }
    return null;
  }, []);

  // --- 旧版 A* 算法 (用于性能对比) ---
  const findPathAStarLegacy = useCallback((
    startId: string, 
    endId: string, 
    currentGrid: Record<string, Cell>,
    currentTrains: Train[] = []
  ): string[] | null => {
    if (!currentGrid[startId] || !currentGrid[endId]) return null;

    // 简化环境检查，仅用于纯算法性能对比
    const startNode = parseCellId(startId);
    const endNode = parseCellId(endId);

    const openSet = new PriorityQueueLegacy<AStarNode>(); // 使用旧队列
    const startH = getHeuristic(startNode.x, startNode.y, endNode.x, endNode.y);
    openSet.push({ 
      id: startId, 
      x: startNode.x, 
      y: startNode.y, 
      g: 0, 
      h: startH, 
      f: startH 
    });

    const visited = new Map<string, number>();
    visited.set(startId, 0);

    while (!openSet.isEmpty()) {
      const current = openSet.pop()!;
      if (current.id === endId) {
        const path: string[] = [];
        let curr: AStarNode | undefined = current;
        while (curr) {
          path.unshift(curr.id);
          curr = curr.parent;
        }
        return path;
      }

      const neighbors = [
        { nx: current.x, ny: current.y - 1 },
        { nx: current.x, ny: current.y + 1 },
        { nx: current.x - 1, ny: current.y },
        { nx: current.x + 1, ny: current.y }
      ];

      for (const { nx, ny } of neighbors) {
        const nid = getCellId(nx, ny);
        const neighborCell = currentGrid[nid];

        if (neighborCell && (neighborCell.type === 'track' || neighborCell.type === 'station')) {
          const newG = current.g + 1;
          if (!visited.has(nid) || newG < visited.get(nid)!) {
            visited.set(nid, newG);
            const h = getHeuristic(nx, ny, endNode.x, endNode.y);
            openSet.push({
              id: nid, x: nx, y: ny, g: newG, h: h, f: newG + h, parent: current
            });
          }
        }
      }
    }
    return null;
  }, []);

  // --- 性能测试函数 ---
  const runAlgorithmBenchmark = useCallback(() => {
    setIsBenchmarking(true);
    setTimeout(() => {
      const iterations = 100; // 次数减少，但单次计算量增大
      
      // --- 构造一个虚拟的大规模测试环境 (100x100) ---
      // 这能模拟真实大地图场景，凸显算法差异
      const BENCH_W = 100;
      const BENCH_H = 100;
      const benchGrid: Record<string, Cell> = {};
      
      // 初始化网格
      for(let y=0; y<BENCH_H; y++) {
        for(let x=0; x<BENCH_W; x++) {
          const id = `${x}-${y}`;
          benchGrid[id] = { x, y, type: 'track', id };
        }
      }
      
      // 添加随机障碍物 (模拟复杂地形，强制A*搜索更多节点)
      for(let i=0; i<2000; i++) {
        const x = Math.floor(Math.random() * BENCH_W);
        const y = Math.floor(Math.random() * BENCH_H);
        const id = `${x}-${y}`;
        if (benchGrid[id]) benchGrid[id].type = 'empty'; // 设为不可通行
      }

      // 生成起终点
      const testCases: {fromId: string, toId: string}[] = [];
      for(let i=0; i<iterations; i++) {
        // 确保起终点距离较远
        const x1 = Math.floor(Math.random() * (BENCH_W/3));
        const y1 = Math.floor(Math.random() * BENCH_H);
        const x2 = Math.floor(BENCH_W - 1 - Math.random() * (BENCH_W/3));
        const y2 = Math.floor(Math.random() * BENCH_H);
        
        testCases.push({ fromId: `${x1}-${y1}`, toId: `${x2}-${y2}` });
      }

      // 2. 测试旧算法
      const startLegacy = performance.now();
      for(const {fromId, toId} of testCases) {
        findPathAStarLegacy(fromId, toId, benchGrid, []);
      }
      const endLegacy = performance.now();
      const legacyTime = endLegacy - startLegacy;

      // 3. 测试新算法
      const startNew = performance.now();
      for(const {fromId, toId} of testCases) {
        findPathAStar(fromId, toId, benchGrid, []);
      }
      const endNew = performance.now();
      const newTime = endNew - startNew;

      // 4. 计算结果
      const improvement = legacyTime > 0 ? ((legacyTime - newTime) / legacyTime * 100).toFixed(1) : "0.0";
      const speedup = newTime > 0 ? (legacyTime / newTime).toFixed(1) : "1.0";

      setBenchmarkResult({
        legacyTime,
        newTime,
        iterations,
        improvement: `${improvement}% (快 ${speedup}x)`
      });
      setIsBenchmarking(false);
    }, 100);
  }, [findPathAStar, findPathAStarLegacy]);

  const isPathClear = useCallback((
    pathSegment: string[], 
    myTrainId: string, 
    allTrains: Train[],
    reservedCells: Set<string>
  ): boolean => {
    for (const cellId of pathSegment) {
      if (reservedCells.has(cellId)) return false;

      const occupier = allTrains.find(t => {
        if (t.id === myTrainId) return false;
        const tPos = t.path[t.currentPathIndex];
        
        if (tPos === cellId) return true;
        
        if (t.state === 'moving' && t.currentPathIndex + 1 < t.path.length) {
          if (t.path[t.currentPathIndex + 1] === cellId) return true;
        }
        return false;
      });

      if (occupier) return false;
    }
    return true;
  }, []);

  const dispatchTrain = useCallback((
    fromName: string, 
    toName: string, 
    priority: number, 
    dispatchType: 'manual' | 'schedule' | 'batch',
    isCyclic: boolean = false
  ) => {
    if (fromName === toName) return; 

    const stationsList = (Object.values(gridRef.current) as Cell[]).filter(c => c.type === 'station');
    const from = stationsList.find(s => s.stationName === fromName);
    const to = stationsList.find(s => s.stationName === toName);
    
    if (from && to) {
      const path = findPathAStar(from.id, to.id, gridRef.current, trainsRef.current);
      if (path) {
        const trainId = `${dispatchType.slice(0,3)}-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`;
        const typePrefix = dispatchType === 'manual' ? '临' : (dispatchType === 'batch' ? '批' : 'T');
        const newTrain: Train = {
          id: trainId,
          name: `${typePrefix}-${trainId.slice(-4)}`,
          fromId: from.id,
          toId: to.id,
          path,
          currentPathIndex: 0,
          progress: 0,
          state: 'waiting',
          priority: priority,
          type: dispatchType,
          color: getUniqueColor(trainsRef.current),
          waitingTime: 0,
          isCyclic: isCyclic,
          dwellRemaining: 0
        };
        setTrains(prev => [...prev, newTrain]);
        
        setHistory(prev => [{
          id: `hist-${Date.now()}-${Math.random()}`,
          time: timeRef.current,
          from: fromName,
          to: toName,
          priority: priority,
          type: dispatchType,
          isCyclic: isCyclic
        }, ...prev]);

      }
    }
  }, [findPathAStar, getUniqueColor]);

  // --- 一键批量发车功能 ---
  const batchDispatchTrains = useCallback(() => {
    const stationsList = (Object.values(gridRef.current) as Cell[]).filter(c => c.type === 'station');
    if (stationsList.length < 2) {
      alert("至少需要2个车站才能发车！");
      return;
    }

    const stationNames = stationsList.map(s => s.stationName!);
    const queue: {from: string, to: string, dispatchAt: number, priority: number, isCyclic: boolean}[] = [];
    
    for (let i = 0; i < batchCount; i++) {
      // 随机选择起终点
      const fromIdx = Math.floor(Math.random() * stationNames.length);
      let toIdx = Math.floor(Math.random() * stationNames.length);
      while (toIdx === fromIdx) {
        toIdx = Math.floor(Math.random() * stationNames.length);
      }
      
      queue.push({
        from: stationNames[fromIdx],
        to: stationNames[toIdx],
        dispatchAt: timeRef.current + i * batchDelay,
        priority: Math.floor(Math.random() * 10) + 1,
        isCyclic: batchIsCyclic
      });
    }
    
    batchQueueRef.current = queue;
    
    // 如果仿真没有运行，自动启动
    if (!isRunning) {
      setIsRunning(true);
    }
  }, [batchCount, batchDelay, batchIsCyclic, isRunning]);

  // 快速测试：随机发送指定数量的车辆（立即发车）
  const quickBatchDispatch = useCallback((count: number, cyclic: boolean = true) => {
    const stationsList = (Object.values(gridRef.current) as Cell[]).filter(c => c.type === 'station');
    if (stationsList.length < 2) return;

    const stationNames = stationsList.map(s => s.stationName!);
    
    for (let i = 0; i < count; i++) {
      const fromIdx = Math.floor(Math.random() * stationNames.length);
      let toIdx = Math.floor(Math.random() * stationNames.length);
      while (toIdx === fromIdx) {
        toIdx = Math.floor(Math.random() * stationNames.length);
      }
      
      setTimeout(() => {
        dispatchTrain(
          stationNames[fromIdx], 
          stationNames[toIdx], 
          Math.floor(Math.random() * 10) + 1, 
          'batch',
          cyclic
        );
      }, i * 100); // 间隔100ms发车，避免同时创建太多
    }
    
    if (!isRunning) {
      setIsRunning(true);
    }
  }, [dispatchTrain, isRunning]);

  // --- AI ---
  const generateScheduleWithAI = async () => {
    if (!process.env.API_KEY || !aiPrompt) return;
    setIsAiLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const stationNames = (Object.values(grid) as Cell[])
        .filter(c => c.type === 'station')
        .map(c => c.stationName)
        .join(", ");

      const systemInstruction = `
        You are a train scheduler assistant.
        The current map has these stations: ${stationNames}.
        The current simulation time is ${time}.
        
        The user will describe a traffic scenario in Chinese or English.
        Generate a JSON schedule.
        Start scheduling trains from time = ${time + 5} or later.
        Priority 1 (Low) to 10 (High).
        IMPORTANT: Use EXACT station names from the provided list.
      `;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: aiPrompt,
        config: {
          systemInstruction: systemInstruction,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                timeOffset: { type: Type.INTEGER, description: "How many seconds from NOW to start." },
                from: { type: Type.STRING, description: "Exact station name from the list." },
                to: { type: Type.STRING, description: "Exact station name from the list." },
                priority: { type: Type.INTEGER, description: "Priority 1-10" }
              },
              required: ["timeOffset", "from", "to", "priority"]
            }
          }
        }
      });

      const generatedData = JSON.parse(response.text);
      
      const newScheduleItems: ScheduleItem[] = generatedData.map((item: any, idx: number) => ({
        id: `sched-ai-${Date.now()}-${idx}`,
        time: time + item.timeOffset,
        from: item.from,
        to: item.to,
        priority: item.priority,
        dispatched: false
      }));

      setSchedule(prev => [...prev, ...newScheduleItems].sort((a, b) => a.time - b.time));
      setAiPrompt("");
    } catch (e) {
      console.error("AI Error:", e);
      alert("生成时刻表失败，请重试。");
    } finally {
      setIsAiLoading(false);
    }
  };

  // --- Simulation Loop ---

  useEffect(() => {
    if (!isRunning) return;

    const actualTickRate = Math.max(20, BASE_TICK_RATE_MS / simulationSpeed);

    const tick = setInterval(() => {
      const currentTime = timeRef.current + 1;
      setTime(currentTime);
      
      // 处理批量发车队列
      const batchQueue = batchQueueRef.current;
      const toDispatch = batchQueue.filter(item => item.dispatchAt <= currentTime);
      if (toDispatch.length > 0) {
        toDispatch.forEach(item => {
          dispatchTrain(item.from, item.to, item.priority, 'batch', item.isCyclic);
        });
        batchQueueRef.current = batchQueue.filter(item => item.dispatchAt > currentTime);
      }
      
      // 处理计划发车
      const currentSched = scheduleRef.current;
      const updatedSched = currentSched.map(item => {
        if (!item.dispatched && item.time <= currentTime) {
          dispatchTrain(item.from, item.to, item.priority, 'schedule');
          return { ...item, dispatched: true };
        }
        return item;
      });
      
      if (updatedSched.some((s, i) => s.dispatched !== currentSched[i].dispatched)) {
        setSchedule(updatedSched);
      }

      // 处理列车
      let activeTrains = [...trainsRef.current];
      const globalReservations = new Set<string>();
      
      activeTrains.sort((a, b) => b.priority - a.priority);

      const nextTrainsState = activeTrains.map(train => {
        let { currentPathIndex, progress, state, path, waitingTime, dwellRemaining } = train;
        
        // --- DWELLING ---
        if (state === 'dwelling') {
          if (dwellRemaining > 0) {
            return { ...train, dwellRemaining: dwellRemaining - 1 };
          }
          const newFrom = train.toId;
          const newTo = train.fromId;
          
          const returnPath = findPathAStar(
            newFrom, 
            newTo, 
            gridRef.current, 
            trainsRef.current, 
            train.id
          );
          
          if (returnPath && returnPath.length > 0) {
            return {
              ...train,
              fromId: newFrom,
              toId: newTo,
              path: returnPath,
              currentPathIndex: 0,
              progress: 0,
              state: 'waiting' as const,
              waitingTime: 0,
              dwellRemaining: 0
            };
          } else {
            return { ...train, dwellRemaining: 10 };
          }
        }

        // --- ARRIVED ---
        if (state === 'arrived') {
          if (train.isCyclic) {
            return { ...train, state: 'dwelling' as const, dwellRemaining: 30 };
          }
          return train; 
        }

        if (!path || !path[currentPathIndex]) {
          return { ...train, state: 'arrived' as const };
        }

        const currentCellId = path[currentPathIndex];
        
        // --- MOVING -> WAITING ---
        if (state === 'moving') {
          waitingTime = 0;
          progress += MOVEMENT_SPEED * simulationSpeed;
          if (progress >= 1.0) {
            progress = 0;
            currentPathIndex++;
            state = 'waiting';
          }
        }

        // --- WAITING ---
        if (state === 'waiting') {
          if (currentPathIndex >= path.length - 1) {
            return { ...train, state: 'arrived' as const, progress: 0, currentPathIndex, waitingTime: 0 };
          }

          const segmentEndIdx = Math.min(path.length, currentPathIndex + RESERVATION_LOOKAHEAD);
          let segmentToReserve = path.slice(currentPathIndex + 1, segmentEndIdx);
          
          const blocked = !isPathClear(segmentToReserve, train.id, trainsRef.current, globalReservations);

          if (blocked) {
            const strictReroutePath = findPathAStar(
              path[currentPathIndex], 
              train.toId, 
              gridRef.current, 
              trainsRef.current,
              train.id,
              globalReservations,
              true
            );

            if (strictReroutePath && strictReroutePath.length > 0) {
              path = [...path.slice(0, currentPathIndex + 1), ...strictReroutePath.slice(1)];
              
              const newSegmentEndIdx = Math.min(path.length, currentPathIndex + RESERVATION_LOOKAHEAD);
              segmentToReserve = path.slice(currentPathIndex + 1, newSegmentEndIdx);
            } 
            else {
              if (waitingTime % 10 === 0) {
                const softReroutePath = findPathAStar(
                  path[currentPathIndex], 
                  train.toId, 
                  gridRef.current, 
                  trainsRef.current,
                  train.id,
                  globalReservations,
                  false
                );
                if (softReroutePath && softReroutePath.length > 0) {
                  path = [...path.slice(0, currentPathIndex + 1), ...softReroutePath.slice(1)];
                }
              }
            }
          }

          if (isPathClear(segmentToReserve, train.id, trainsRef.current, globalReservations)) {
            segmentToReserve.forEach(cid => globalReservations.add(cid));
            state = 'moving';
            waitingTime = 0;
          } else {
            globalReservations.add(currentCellId);
            waitingTime++;
          }
        } 
        else if (state === 'moving') {
          const segmentEndIdx = Math.min(path.length, currentPathIndex + RESERVATION_LOOKAHEAD);
          const segmentToKeep = path.slice(currentPathIndex + 1, segmentEndIdx);
          segmentToKeep.forEach(cid => globalReservations.add(cid));
        }

        return { ...train, currentPathIndex, progress, state, path, waitingTime };
      });

      const runningTrains = nextTrainsState.filter(t => t.state !== 'arrived' || t.isCyclic);
      
      if (runningTrains.length > 1 && runningTrains.every(t => t.state === 'waiting' && t.waitingTime > 20)) {
        setGlobalDeadlockWarning(true);
      } else {
        setGlobalDeadlockWarning(false);
      }

      setTrains(runningTrains);

    }, actualTickRate);

    return () => clearInterval(tick);
  }, [isRunning, simulationSpeed, findPathAStar, isPathClear, dispatchTrain]); 


  // --- Event Handlers ---

  const modifyCell = (x: number, y: number, action: 'track' | 'station' | 'erase') => {
    setGrid(prev => {
      const id = getCellId(x, y);
      const cell = prev[id];
      if (!cell) return prev;

      if (action === 'erase') {
        if (cell.type === 'empty') return prev;
        return { ...prev, [id]: { ...cell, type: 'empty', stationName: undefined } };
      }

      if (action === 'track') {
        if (cell.type === 'track') return prev;
        return { ...prev, [id]: { ...cell, type: 'track', stationName: undefined } };
      }

      if (action === 'station') {
        if (cell.type === 'station') return prev;
        return { ...prev, [id]: { ...cell, type: 'station', stationName: `站${x}-${y}` } };
      }
      return prev;
    });
  };

  const handleCellMouseDown = (x: number, y: number, e: React.MouseEvent) => {
    if (isRunning) return;
    if (e.button === 2) e.preventDefault();
    isMouseDown.current = true;
    const isRight = e.button === 2;
    if (isRight) {
      dragAction.current = 'erase';
      modifyCell(x, y, 'erase');
    } else {
      if (editMode === 'track') {
        dragAction.current = 'track';
        modifyCell(x, y, 'track');
      } else {
        dragAction.current = null;
        modifyCell(x, y, 'station');
      }
    }
  };

  const handleCellMouseEnter = (x: number, y: number) => {
    if (isRunning || !isMouseDown.current || !dragAction.current) return;
    modifyCell(x, y, dragAction.current);
  };

  const handleCellDoubleClick = (x: number, y: number) => {
    if (isRunning) return;
    const id = getCellId(x, y);
    const cell = grid[id];
    if (cell.type === 'station') {
      const newName = prompt("重命名车站:", cell.stationName);
      if (newName && newName.trim() !== "") {
        setGrid(prev => ({
          ...prev,
          [id]: { ...cell, stationName: newName.trim() }
        }));
      }
    }
  };

  // Preview logic
  const stations = useMemo(() => (Object.values(grid) as Cell[]).filter(c => c.type === 'station'), [grid]);
  
  useEffect(() => {
    if (!dispatchFrom || !dispatchTo || dispatchFrom === dispatchTo) {
      setPreviewPath([]);
      return;
    }
    const s1 = stations.find(s => s.stationName === dispatchFrom);
    const s2 = stations.find(s => s.stationName === dispatchTo);
    if (s1 && s2) {
      const path = findPathAStar(s1.id, s2.id, grid, trains);
      setPreviewPath(path || []);
    } else {
      setPreviewPath([]);
    }
  }, [dispatchFrom, dispatchTo, grid, stations, trains, findPathAStar]);

  const handleManualDispatch = () => {
    dispatchTrain(dispatchFrom, dispatchTo, dispatchPriority, 'manual', dispatchIsCyclic);
  };

  // --- Render Helpers ---

  const trackLines = useMemo(() => {
    const lines: React.ReactElement[] = [];
    (Object.values(grid) as Cell[]).forEach(cell => {
      if (cell.type === 'empty') return;
      const neighbors = [{ dx: 1, dy: 0 }, { dx: 0, dy: 1 }];
      neighbors.forEach(({ dx, dy }) => {
        const nx = cell.x + dx;
        const ny = cell.y + dy;
        const nid = getCellId(nx, ny);
        const neighbor = grid[nid];
        if (neighbor && neighbor.type !== 'empty') {
          lines.push(
            <line
              key={`${cell.id}-${nid}`}
              x1={cell.x * CELL_SIZE + CELL_SIZE/2}
              y1={cell.y * CELL_SIZE + CELL_SIZE/2}
              x2={nx * CELL_SIZE + CELL_SIZE/2}
              y2={ny * CELL_SIZE + CELL_SIZE/2}
              stroke="#475569" 
              strokeWidth="4"
              strokeLinecap="round"
            />
          );
        }
      });
    });
    return lines;
  }, [grid]);

  const activeTrainPaths = useMemo(() => {
    return trains.map(t => {
      if (!t.path || t.path.length === 0) return null;
      const remainingPath = t.path.slice(Math.max(0, t.currentPathIndex));
      if (remainingPath.length < 2) return null;
      
      const points = remainingPath.map(id => {
        const { x, y } = parseCellId(id);
        return `${x * CELL_SIZE + CELL_SIZE/2},${y * CELL_SIZE + CELL_SIZE/2}`;
      }).join(" ");

      return (
        <polyline
          key={`path-${t.id}`}
          points={points}
          fill="none"
          stroke={t.color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeOpacity="0.15"
        />
      );
    });
  }, [trains]);

  const previewPolyline = useMemo(() => {
    if (previewPath.length < 2) return null;
    const points = previewPath.map(id => {
      const { x, y } = parseCellId(id);
      return `${x * CELL_SIZE + CELL_SIZE/2},${y * CELL_SIZE + CELL_SIZE/2}`;
    }).join(" ");
    return (
      <polyline
        points={points}
        fill="none"
        stroke="#eab308"
        strokeWidth="3"
        strokeDasharray="6 4"
        strokeLinecap="round"
        strokeOpacity="0.8"
        className="animate-pulse"
      />
    );
  }, [previewPath]);

  const hoveredTrainInfo = useMemo(() => {
    if (!hoveredTrainId) return null;
    const train = trains.find(t => t.id === hoveredTrainId);
    if (!train) return null;
    
    if (!train.path || !train.path[train.currentPathIndex]) return null;

    const currentPos = parseCellId(train.path[train.currentPathIndex]);
    const nextPos = train.currentPathIndex < train.path.length - 1 
      ? parseCellId(train.path[train.currentPathIndex + 1]) 
      : currentPos;
    const screenX = (currentPos.x + (nextPos.x - currentPos.x) * train.progress) * CELL_SIZE + CELL_SIZE/2;
    const screenY = (currentPos.y + (nextPos.y - currentPos.y) * train.progress) * CELL_SIZE + CELL_SIZE/2;
    const fromStation = stations.find(s => s.id === train.fromId)?.stationName || "Unknown";
    const toStation = stations.find(s => s.id === train.toId)?.stationName || "Unknown";
    return { ...train, screenX, screenY, fromStation, toStation };
  }, [hoveredTrainId, trains, stations]);

  const pendingTrains = schedule.filter(s => !s.dispatched);
  const pendingBatchCount = batchQueueRef.current.length;
  
  const uniqueHistory = useMemo(() => {
    const seen = new Set<string>();
    return history.filter(item => {
      if (historyFilterFrom && item.from !== historyFilterFrom) return false;
      if (historyFilterTo && item.to !== historyFilterTo) return false;

      const key = `${item.from}->${item.to}`;
      if (seen.has(key)) return false;
      
      seen.add(key);
      return true;
    });
  }, [history, historyFilterFrom, historyFilterTo]);

  return (
    <div className="flex flex-row h-full bg-slate-900 text-slate-100 font-sans select-none">
      {/* LEFT PANEL: MAP */}
      <div className="flex-1 relative flex flex-col h-full bg-slate-950 overflow-hidden">
        
        {/* Header Indicators */}
        <div className="absolute top-4 left-6 flex items-center space-x-4 bg-slate-900/80 backdrop-blur-md p-3 rounded-xl border border-slate-800 shadow-xl z-20 pointer-events-none">
          <div className="flex flex-col">
            <span className="text-xs text-slate-400 font-medium">仿真时间</span>
            <span className="text-xl font-mono text-cyan-400">{time}s</span>
          </div>
          <div className="h-8 w-px bg-slate-700"></div>
          <div className="flex flex-col">
            <span className="text-xs text-slate-400 font-medium">在途列车</span>
            <span className="text-xl font-mono text-emerald-400">{trains.length}</span>
          </div>
          <div className="h-8 w-px bg-slate-700"></div>
          <div className="flex flex-col">
            <span className="text-xs text-slate-400 font-medium">仿真速度</span>
            <span className="text-xl font-mono text-orange-400">{simulationSpeed}x</span>
          </div>
          {globalDeadlockWarning && (
            <>
              <div className="h-8 w-px bg-slate-700"></div>
              <div className="flex items-center space-x-2 text-rose-500 animate-pulse font-bold">
                <span className="text-2xl">⚠</span>
                <span>系统拥堵</span>
              </div>
            </>
          )}
        </div>

        <div className="flex-1 overflow-auto w-full h-full p-6">
          <div className="min-w-min min-h-min flex items-center justify-center">
            <div 
              className="relative bg-slate-900/50 shadow-2xl border border-slate-800 rounded-lg backdrop-blur-sm transition-all duration-300"
          style={{ width: GRID_W * CELL_SIZE, height: GRID_H * CELL_SIZE }}
          onContextMenu={(e) => e.preventDefault()}
        >
          <div className="absolute inset-0 opacity-10 pointer-events-none" 
            style={{ 
              backgroundImage: `linear-gradient(#475569 1px, transparent 1px), linear-gradient(90deg, #475569 1px, transparent 1px)`,
              backgroundSize: `${CELL_SIZE}px ${CELL_SIZE}px`
            }}
          ></div>

          <div className={`absolute inset-0 grid z-10 ${isRunning ? 'pointer-events-none' : 'pointer-events-auto'}`} style={{ 
            gridTemplateColumns: `repeat(${GRID_W}, 1fr)`,
            gridTemplateRows: `repeat(${GRID_H}, 1fr)`
          }}>
            {Array.from({ length: GRID_H * GRID_W }).map((_, i) => {
              const x = i % GRID_W;
              const y = Math.floor(i / GRID_W);
              const id = getCellId(x, y);
              return (
                <div 
                  key={id}
                  className={`border border-transparent hover:border-white/10`}
                  style={{ cursor: isRunning ? 'default' : 'crosshair' }}
                  onMouseDown={(e) => handleCellMouseDown(x, y, e)}
                  onMouseEnter={() => handleCellMouseEnter(x, y)}
                  onDoubleClick={() => handleCellDoubleClick(x, y)}
                />
              );
            })}
          </div>

          <svg className="absolute inset-0 pointer-events-none z-0" width="100%" height="100%">
            {trackLines}
            {activeTrainPaths}
            {previewPolyline}

            {stations.map(s => (
              <g key={s.id}>
                <rect x={s.x * CELL_SIZE + 2} y={s.y * CELL_SIZE + 4} width={CELL_SIZE - 4} height={CELL_SIZE - 8} fill="#000" opacity="0.3" rx="4" />
                <rect x={s.x * CELL_SIZE + 2} y={s.y * CELL_SIZE + 2} width={CELL_SIZE - 4} height={CELL_SIZE - 8} fill="#3b82f6" stroke="#60a5fa" strokeWidth="1" rx="4" />
                <g transform={`translate(${s.x * CELL_SIZE + CELL_SIZE/2}, ${s.y * CELL_SIZE - 8})`}>
                  <rect x="-24" y="-8" width="48" height="14" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
                  <text x="0" y="2" textAnchor="middle" fill="#e2e8f0" fontSize="9" fontWeight="bold" dominantBaseline="middle">{s.stationName}</text>
                </g>
              </g>
            ))}
            
            {trains.map(t => {
              if (!t.path[t.currentPathIndex]) return null;
              const currentPos = parseCellId(t.path[t.currentPathIndex]);
              const nextPos = t.currentPathIndex < t.path.length - 1 
                ? parseCellId(t.path[t.currentPathIndex + 1]) 
                : currentPos;
              const screenX = (currentPos.x + (nextPos.x - currentPos.x) * t.progress) * CELL_SIZE + CELL_SIZE/2;
              const screenY = (currentPos.y + (nextPos.y - currentPos.y) * t.progress) * CELL_SIZE + CELL_SIZE/2;
              const isHovered = hoveredTrainId === t.id;
              const isStuck = t.waitingTime > DEADLOCK_THRESHOLD_TICKS;

              return (
                <g 
                  key={t.id} 
                  style={{ transition: `all ${BASE_TICK_RATE_MS / simulationSpeed}ms linear` }}
                  className="pointer-events-auto cursor-help"
                  onMouseEnter={() => setHoveredTrainId(t.id)}
                  onMouseLeave={() => setHoveredTrainId(null)}
                >
                  {isStuck && (
                    <text x={screenX} y={screenY - 24} textAnchor="middle" fontSize="16" className="animate-bounce">⚠️</text>
                  )}

                  {isHovered && <circle cx={screenX} cy={screenY} r={CELL_SIZE * 0.8} fill={t.color} opacity="0.3" className="animate-pulse" />}
                  <circle cx={screenX} cy={screenY} r={CELL_SIZE * 0.5} fill={t.color} opacity="0.2" />
                  <circle 
                    cx={screenX} 
                    cy={screenY} 
                    r={CELL_SIZE * 0.35} 
                    fill={t.state === 'waiting' ? '#ef4444' : (t.state === 'dwelling' ? '#f59e0b' : t.color)}
                    stroke="white"
                    strokeWidth={isHovered ? "3" : "2"}
                    className="shadow-lg drop-shadow-md"
                  />
                  <text x={screenX} y={screenY - 14} textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" filter="drop-shadow(0px 1px 1px rgba(0,0,0,0.8))">
                    {t.isCyclic ? '↻' : ''}{t.name}
                  </text>
                </g>
              );
            })}
          </svg>
          
          {hoveredTrainInfo && (
            <div 
              className="absolute z-50 pointer-events-none flex flex-col gap-1 p-2 bg-slate-800/90 backdrop-blur text-slate-100 rounded-lg shadow-2xl border border-slate-600 text-[10px]"
              style={{
                left: hoveredTrainInfo.screenX,
                top: hoveredTrainInfo.screenY + 20,
                transform: 'translate(-50%, 0)',
                minWidth: '120px'
              }}
            >
              <div className="font-bold text-xs text-white border-b border-slate-600 pb-1 mb-1 flex justify-between items-center">
                <span>{hoveredTrainInfo.name} {hoveredTrainInfo.isCyclic && <span className="text-blue-400 ml-1">(循环)</span>}</span>
                <span className="px-1.5 rounded-full" style={{backgroundColor: hoveredTrainInfo.color}}></span>
              </div>
              <div className="flex justify-between">
                <span>状态:</span> 
                <span className={hoveredTrainInfo.state === 'waiting' ? 'text-red-400' : (hoveredTrainInfo.state === 'dwelling' ? 'text-yellow-400' : 'text-emerald-400')}>
                  {hoveredTrainInfo.state === 'waiting' ? '等待中' : (hoveredTrainInfo.state === 'dwelling' ? '折返中' : '运行中')}
                </span>
              </div>
              {hoveredTrainInfo.state === 'dwelling' && (
                <div className="flex justify-between"><span>折返剩余:</span> <span className="text-yellow-400">{hoveredTrainInfo.dwellRemaining}s</span></div>
              )}
              <div className="flex justify-between"><span>等待:</span> <span className={`${hoveredTrainInfo.waitingTime > 20 ? 'text-rose-400 font-bold' : 'text-slate-400'}`}>{hoveredTrainInfo.waitingTime} Ticks</span></div>
              <div className="flex justify-between"><span>优先级:</span> <span className="text-orange-300 font-mono">P{hoveredTrainInfo.priority}</span></div>
              <div className="flex justify-between"><span>路线:</span> <span className="text-slate-400">{hoveredTrainInfo.fromStation} → {hoveredTrainInfo.toStation}</span></div>
            </div>
          )}
        </div>
          </div>
        </div>
        
        <div className="absolute bottom-6 left-6 flex space-x-4 text-xs text-slate-500 bg-slate-900/80 p-2 rounded-lg border border-slate-800 z-20 pointer-events-none">
          <div className="flex items-center space-x-1"><span className="w-3 h-3 bg-blue-500 rounded"></span><span>车站</span></div>
          <div className="flex items-center space-x-1"><span className="w-3 h-3 bg-slate-600 rounded"></span><span>轨道</span></div>
          <div className="border-l border-slate-700 pl-2">
            操作: 左键拖拽铺路/点击建站 | 右键拖拽擦除 | 悬浮查看详情 | 双击重命名
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col shadow-2xl z-20">
        <div className="p-6 border-b border-slate-800 bg-slate-900">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
            列车调度中心
          </h1>
          <p className="text-xs text-slate-500 mt-1">Advanced Train Dispatch System v2.0</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* 控制按钮 */}
          <div className="flex space-x-2">
            <button 
              onClick={() => setIsRunning(!isRunning)}
              className={`flex-1 py-2 px-4 rounded-lg font-bold text-white shadow-lg transition-all active:scale-95 text-sm
                ${isRunning ? 'bg-rose-600 hover:bg-rose-700' : 'bg-emerald-600 hover:bg-emerald-700'}`}
            >
              {isRunning ? "⏹ 停止" : "▶ 启动"}
            </button>
            <button 
              onClick={clearAllTrains}
              className="px-3 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg font-bold shadow-lg text-sm active:scale-95"
              title="清除所有运行中的列车"
            >
              🗑
            </button>
            <button 
              onClick={resetSystem}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg font-bold shadow-lg text-sm active:scale-95"
            >
              ↺ 重置
            </button>
            <button 
              onClick={clearTracks}
              className="px-4 py-2 bg-red-900/50 hover:bg-red-800/50 text-red-200 border border-red-800 rounded-lg font-bold shadow-lg text-sm active:scale-95"
              title="清除所有轨道(保留车站)"
            >
              🧹 清轨
            </button>
          </div>
          
          {/* 速度控制 */}
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-slate-400">仿真速度</span>
              <span className="text-sm font-mono text-orange-400">{simulationSpeed}x</span>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="5" 
              step="0.5"
              value={simulationSpeed} 
              onChange={(e) => setSimulationSpeed(Number(e.target.value))} 
              className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>0.5x</span>
              <span>1x</span>
              <span>2x</span>
              <span>3x</span>
              <span>5x</span>
            </div>
          </div>

          {/* 编辑模式 */}
          <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg text-xs text-slate-400 border border-slate-700">
            <button className={`flex-1 py-1 rounded ${editMode === 'track' ? 'bg-slate-600 text-white' : 'hover:bg-slate-700'}`} onClick={() => setEditMode('track')}>
              轨道模式
            </button>
            <button className={`flex-1 py-1 rounded ${editMode === 'station' ? 'bg-blue-600 text-white' : 'hover:bg-slate-700'}`} onClick={() => setEditMode('station')}>
              车站模式
            </button>
          </div>

          {/* 一键批量发车 */}
          <div className="bg-gradient-to-br from-emerald-900/40 to-teal-900/40 rounded-xl p-4 border border-emerald-500/30">
            <h2 className="text-xs font-bold text-emerald-300 uppercase tracking-wider mb-3 flex items-center gap-2">
              🚄 一键批量测试
            </h2>
            
            {/* 快捷按钮 */}
            <div className="grid grid-cols-4 gap-2 mb-3">
              <button 
                onClick={() => quickBatchDispatch(5, batchIsCyclic)}
                className="bg-emerald-600 hover:bg-emerald-500 text-white py-2 rounded-lg text-xs font-bold shadow-lg active:scale-95"
              >
                5辆
              </button>
              <button 
                onClick={() => quickBatchDispatch(10, batchIsCyclic)}
                className="bg-emerald-600 hover:bg-emerald-500 text-white py-2 rounded-lg text-xs font-bold shadow-lg active:scale-95"
              >
                10辆
              </button>
              <button 
                onClick={() => quickBatchDispatch(20, batchIsCyclic)}
                className="bg-emerald-600 hover:bg-emerald-500 text-white py-2 rounded-lg text-xs font-bold shadow-lg active:scale-95"
              >
                20辆
              </button>
              <button 
                onClick={() => quickBatchDispatch(50, batchIsCyclic)}
                className="bg-teal-600 hover:bg-teal-500 text-white py-2 rounded-lg text-xs font-bold shadow-lg active:scale-95"
              >
                50辆
              </button>
            </div>
            
            {/* 高级设置 */}
            <div className="space-y-2 border-t border-emerald-700/50 pt-3">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-300 w-16">数量:</span>
                <input 
                  type="number" 
                  min="1" 
                  max="100" 
                  value={batchCount} 
                  onChange={(e) => setBatchCount(Math.min(100, Math.max(1, Number(e.target.value))))}
                  className="flex-1 bg-slate-900 border border-slate-600 text-xs rounded p-1.5 text-slate-200 outline-none w-16"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-300 w-16">间隔:</span>
                <input 
                  type="number" 
                  min="1" 
                  max="50" 
                  value={batchDelay} 
                  onChange={(e) => setBatchDelay(Math.min(50, Math.max(1, Number(e.target.value))))}
                  className="flex-1 bg-slate-900 border border-slate-600 text-xs rounded p-1.5 text-slate-200 outline-none w-16"
                />
                <span className="text-[10px] text-slate-500">Tick</span>
              </div>
              <div className="flex items-center gap-2">
                <input 
                  type="checkbox" 
                  id="batchCyclicCheck" 
                  checked={batchIsCyclic} 
                  onChange={(e) => setBatchIsCyclic(e.target.checked)} 
                  className="rounded bg-slate-800 border-slate-600" 
                />
                <label htmlFor="batchCyclicCheck" className="text-xs text-slate-300 select-none cursor-pointer">循环往返</label>
              </div>
              <button 
                onClick={batchDispatchTrains}
                className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white py-2 rounded-lg text-xs font-bold shadow-lg active:scale-95"
              >
                🚀 按设置批量发车
              </button>
            </div>
          </div>

          {/* 单次发车 */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">临时发车指令</h2>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <select className="bg-slate-900 border border-slate-600 text-xs rounded-lg p-2 text-slate-200 outline-none" value={dispatchFrom} onChange={(e) => setDispatchFrom(e.target.value)}>
                  <option value="">始发站...</option>
                  {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                </select>
                <select className="bg-slate-900 border border-slate-600 text-xs rounded-lg p-2 text-slate-200 outline-none" value={dispatchTo} onChange={(e) => setDispatchTo(e.target.value)}>
                  <option value="">终点站...</option>
                  {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <input type="checkbox" id="cyclicCheck" checked={dispatchIsCyclic} onChange={(e) => setDispatchIsCyclic(e.target.checked)} className="rounded bg-slate-800 border-slate-600" />
                <label htmlFor="cyclicCheck" className="text-xs text-slate-300 select-none cursor-pointer">循环发车</label>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">优先级: P{dispatchPriority}</span>
                <input type="range" min="1" max="10" value={dispatchPriority} onChange={(e) => setDispatchPriority(Number(e.target.value))} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"/>
              </div>
              <button onClick={handleManualDispatch} disabled={previewPath.length === 0 || dispatchFrom === dispatchTo} className="w-full bg-orange-600 hover:bg-orange-500 disabled:bg-slate-700 text-white py-2 rounded-lg text-xs font-bold shadow-lg">立即发车</button>
            </div>
          </div>

          {/* 队列状态 */}
          <div className="bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
            <div className="p-3 bg-slate-800/80 border-b border-slate-700 flex justify-between items-center">
              <label className="text-xs font-bold text-yellow-500 uppercase flex items-center gap-1">
                <span>⏳</span> 发车队列
              </label>
              <span className="text-[10px] bg-slate-700 px-2 py-0.5 rounded-full text-slate-300">
                计划:{pendingTrains.length} | 批量:{pendingBatchCount}
              </span>
            </div>
            <div className="max-h-24 overflow-y-auto custom-scrollbar p-2 space-y-1">
              {pendingTrains.length === 0 && pendingBatchCount === 0 && <p className="text-[10px] text-slate-600 text-center py-2">队列空闲</p>}
              {pendingTrains.map(item => (
                <div key={item.id} className="flex justify-between items-center text-[10px] p-2 rounded bg-slate-800 border border-slate-700 text-slate-300">
                  <span>T+{item.time}s | {item.from} → {item.to}</span>
                  <span className="text-yellow-500">计划</span>
                </div>
              ))}
              {pendingBatchCount > 0 && (
                <div className="text-[10px] p-2 rounded bg-emerald-900/30 border border-emerald-700/50 text-emerald-300 text-center">
                  批量发车队列中还有 {pendingBatchCount} 辆待发
                </div>
              )}
            </div>
          </div>

          {/* 历史记录 */}
          <div className="flex-1 flex flex-col min-h-[150px] bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
            <div className="p-3 bg-slate-800/80 border-b border-slate-700 space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-xs font-bold text-slate-400 uppercase">历史路线</label>
                <span className="text-[10px] bg-slate-700 px-2 py-0.5 rounded-full text-slate-300">{uniqueHistory.length}</span>
              </div>
              
              <div className="flex gap-2">
                <select className="flex-1 bg-slate-900 border border-slate-600 text-[10px] rounded p-1 text-slate-200 outline-none" 
                  value={historyFilterFrom} onChange={(e) => setHistoryFilterFrom(e.target.value)}>
                  <option value="">全部始发...</option>
                  {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                </select>
                <select className="flex-1 bg-slate-900 border border-slate-600 text-[10px] rounded p-1 text-slate-200 outline-none" 
                  value={historyFilterTo} onChange={(e) => setHistoryFilterTo(e.target.value)}>
                  <option value="">全部终点...</option>
                  {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                </select>
              </div>
            </div>

            <div className="overflow-y-auto flex-1 p-2 space-y-1 custom-scrollbar">
              {uniqueHistory.length === 0 && <p className="text-[10px] text-slate-600 text-center py-4">无历史记录</p>}
              {uniqueHistory.slice(0, 20).map(item => (
                <div key={item.id} className="flex justify-between items-center text-xs p-2 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-transparent hover:border-slate-600 group transition-all">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-slate-400">P{item.priority}</span>
                      <span className={`text-[9px] px-1 rounded ${item.type === 'manual' ? 'bg-orange-900/50 text-orange-400' : (item.type === 'batch' ? 'bg-emerald-900/50 text-emerald-400' : 'bg-blue-900/50 text-blue-400')}`}>
                        {item.type === 'manual' ? '手动' : (item.type === 'batch' ? '批量' : '计划')}
                      </span>
                      {item.isCyclic && <span className="text-[9px] px-1 rounded bg-purple-900/50 text-purple-400">循环</span>}
                    </div>
                    <span className="text-[10px] mt-0.5">{item.from} ➔ {item.to}</span>
                  </div>
                  <button 
                    onClick={() => dispatchTrain(item.from, item.to, item.priority, 'manual', item.isCyclic)}
                    className="opacity-0 group-hover:opacity-100 bg-emerald-600 hover:bg-emerald-500 text-white px-2 py-1 rounded text-[10px] transition-opacity"
                    title="再次发车"
                  >
                    ↺
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* AI调度 */}
          <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 rounded-xl p-4 border border-indigo-500/30">
            <h2 className="text-xs font-bold text-indigo-300 uppercase tracking-wider mb-2">AI 智能调度</h2>
            <textarea className="w-full bg-slate-900/80 border border-indigo-500/30 rounded-lg p-2 text-xs text-indigo-100 h-16 outline-none resize-none" placeholder="描述场景..." value={aiPrompt} onChange={(e) => setAiPrompt(e.target.value)} />
            <button onClick={generateScheduleWithAI} disabled={isAiLoading || !aiPrompt} className="w-full mt-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white py-1.5 rounded-lg text-xs font-bold">
              {isAiLoading ? "生成中..." : "生成计划"}
            </button>
          </div>

          {/* 算法性能对比测试 */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">🚀 算法效率评估</h2>
            
            {!benchmarkResult ? (
              <div className="text-center py-2">
                 <p className="text-[10px] text-slate-500 mb-3">
                   对比【二叉堆优化 A*】与【传统数组排序 A*】。
                   将在 100x100 的复杂虚拟网格中执行 100 次长距离搜索。
                 </p>
                 <button 
                   onClick={runAlgorithmBenchmark} 
                   disabled={isBenchmarking}
                   className="w-full bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-200 py-2 rounded-lg text-xs font-bold"
                 >
                   {isBenchmarking ? "正在进行高强度压测..." : "开始大规模性能测试"}
                 </button>
              </div>
            ) : (
              <div className="space-y-2 animate-in fade-in duration-500">
                 <div className="grid grid-cols-2 gap-2 text-center mb-2">
                    <div className="bg-slate-900/80 p-2 rounded border border-red-900/30">
                       <div className="text-[10px] text-slate-500">旧算法耗时</div>
                       <div className="text-sm font-mono text-red-400">{benchmarkResult.legacyTime.toFixed(1)}ms</div>
                    </div>
                    <div className="bg-slate-900/80 p-2 rounded border border-emerald-900/30">
                       <div className="text-[10px] text-slate-500">新算法耗时</div>
                       <div className="text-sm font-mono text-emerald-400">{benchmarkResult.newTime.toFixed(1)}ms</div>
                    </div>
                 </div>
                 <div className="bg-emerald-900/20 border border-emerald-500/30 p-2 rounded text-center">
                    <div className="text-[10px] text-emerald-300 mb-1">性能提升</div>
                    <div className="text-lg font-bold text-emerald-400">{benchmarkResult.improvement}</div>
                 </div>
                 <button 
                   onClick={() => setBenchmarkResult(null)} 
                   className="w-full mt-2 text-[10px] text-slate-500 hover:text-slate-300 underline"
                 >
                   重置测试
                 </button>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
