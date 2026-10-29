"""Queue-aware shortest-path routing simulation for BCSE308P."""
from __future__ import annotations
import argparse, csv, heapq, math, os, random
from pathlib import Path
from dataclasses import dataclass
from collections import deque, defaultdict

@dataclass
class Packet:
    created: int
    route: tuple[str, ...]
    pos: int = 0

class Link:
    def __init__(self, name, src, dst, delay, capacity, buffer):
        self.name, self.src, self.dst = name, src, dst
        self.delay, self.capacity, self.buffer = delay, capacity, buffer
        self.queue = deque(); self.in_flight = []

    def occupancy(self):
        return len(self.queue)

class Network:
    def __init__(self, algorithm, seed, rate, ticks, alpha=2.5, reroute=5, scenario='bottleneck'):
        self.algorithm, self.seed, self.rate, self.ticks = algorithm, seed, rate, ticks
        self.alpha, self.reroute, self.scenario = alpha, reroute, scenario
        self.links = {}
        def add(name, a, b, d, c, q): self.links[name] = Link(name,a,b,d,c,q)
        short_capacity = 1 if scenario == 'bottleneck' else 2
        add('SA','S','A',1,short_capacity,6); add('AD','A','D',1,short_capacity,6); add('DG','D','G',1,short_capacity,6); add('GT','G','T',1,short_capacity,6)
        add('SB','S','B',2,2,8); add('BE','B','E',1,2,8); add('EH','E','H',1,2,8); add('HT','H','T',1,2,8)
        add('SC','S','C',2,3,10); add('CF','C','F',2,3,10); add('FT','F','T',2,3,10)
        add('AE','A','E',3,1,6); add('BD','B','D',3,1,6); add('CE','C','E',3,1,6); add('FH','F','H',3,1,6)
        self.adj = defaultdict(list)
        for l in self.links.values(): self.adj[l.src].append(l)
        self.rng = random.Random(seed); self.delay_sum=0; self.delivered=0
        self.generated=0; self.lost=0; self.route_changes=0; self.last_route=None
        self.sequence=0; self.cached_route=None; self.path_counts=defaultdict(int)
        self.observed_peak=defaultdict(int)

    def shortest_route(self):
        dist={'S':0}; prev={}; heap=[(0,'S')]
        while heap:
            cost,u=heapq.heappop(heap)
            if cost != dist.get(u): continue
            if u == 'T': break
            for l in self.adj[u]:
                if self.algorithm == 'baseline': w=l.delay
                else: w=l.delay*(1+self.alpha*self.observed_peak[l.name]/l.buffer)
                nd=cost+w
                if nd < dist.get(l.dst, math.inf): dist[l.dst]=nd; prev[l.dst]=l.name; heapq.heappush(heap,(nd,l.dst))
        names=[]; u='T'
        while u!='S':
            n=prev[u]; names.append(n); u=self.links[n].src
        return tuple(reversed(names))

    def run(self):
        for t in range(self.ticks):
            for l in self.links.values():
                arrivals=[]
                while l.in_flight and l.in_flight[0][0] <= t: arrivals.append(heapq.heappop(l.in_flight)[2])
                for p in arrivals:
                    p.pos += 1
                    if p.pos == len(p.route): self.delivered += 1; self.delay_sum += t-p.created
                    else:
                        next_link=self.links[p.route[p.pos]]
                        if len(next_link.queue) >= next_link.buffer: self.lost += 1
                        else: next_link.queue.append(p)
                for _ in range(l.capacity):
                    if l.queue:
                        self.sequence += 1
                        heapq.heappush(l.in_flight,(t+l.delay,self.sequence,l.queue.popleft()))
            if self.cached_route is None or t % self.reroute == 0:
                self.cached_route=self.shortest_route()
                self.observed_peak=defaultdict(int)
            arrivals = int(self.rate) + (1 if self.rng.random() < self.rate-int(self.rate) else 0)
            for _ in range(arrivals):
                self.generated += 1; route=self.cached_route
                self.path_counts['-'.join(route)] += 1
                if self.last_route is not None and route != self.last_route: self.route_changes += 1
                self.last_route=route
                first=self.links[route[0]]
                if len(first.queue) >= first.buffer: self.lost += 1
                else: first.queue.append(Packet(t,route))
            for l in self.links.values():
                self.observed_peak[l.name]=max(self.observed_peak[l.name],len(l.queue))
        loss=self.lost/self.generated if self.generated else 0
        pending=sum(len(l.queue)+len(l.in_flight) for l in self.links.values())
        assert self.generated == self.delivered+self.lost+pending, (self.generated,self.delivered,self.lost,pending)
        return {'scenario':self.scenario,'algorithm':self.algorithm,'traffic':self.rate,'seed':self.seed,'ticks':self.ticks,
                'generated':self.generated,'delivered':self.delivered,'lost':self.lost,
                'avg_delay':self.delay_sum/self.delivered if self.delivered else 0,
                'loss_ratio':loss,'throughput':self.delivered/self.ticks,
                'route_changes':self.route_changes,'short_route_packets':self.path_counts['SA-AD-DG-GT'],
                'middle_route_packets':self.path_counts['SB-BE-EH-HT'],
                'lower_route_packets':self.path_counts['SC-CF-FT'],
                'cross_route_packets':sum(v for k,v in self.path_counts.items() if k not in {'SA-AD-DG-GT','SB-BE-EH-HT','SC-CF-FT'}),
                'pending_at_end':pending}

def plot(rows, out):
    import matplotlib.pyplot as plt
    rates=sorted({r['traffic'] for r in rows})
    metrics=[('avg_delay','Average end-to-end delay (ticks)'),('loss_ratio','Packet loss ratio'),('throughput','Delivered throughput (packets/tick)')]
    for key,label in metrics:
        plt.figure(figsize=(7,4.2));
        for alg,style in [('baseline','o-'),('queue_aware','s-')]:
            xs=[]; ys=[]
            for rate in rates:
                vals=[r[key] for r in rows if r['algorithm']==alg and r['traffic']==rate]
                xs.append(rate); ys.append(sum(vals)/len(vals))
            plt.plot(xs,ys,style,label=alg.replace('_',' ').title())
        plt.xlabel('Offered packets per tick'); plt.ylabel(label); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out,key+'.png'),dpi=180); plt.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir'); ap.add_argument('--seeds',type=int,default=10); ap.add_argument('--ticks',type=int,default=5000); ap.add_argument('--alpha',type=float,default=2.5); ap.add_argument('--update-interval',type=int,default=5); ap.add_argument('--scenario',choices=['bottleneck','balanced'],default='bottleneck'); ap.add_argument('--loads',type=float,nargs='+'); ap.add_argument('--log',help='Save the same printed output to a UTF-8 text file'); ap.add_argument('--table',action='store_true',help='Print a compact measured comparison table'); args=ap.parse_args()
    if args.loads is None: args.loads=[1.0,1.2,2.0,3.0] if args.table else [0.5,1.2,2.0]
    if args.output_dir is None: args.output_dir='Review2_Table_Run' if args.table else 'outputs'
    if args.seeds < 1 or args.ticks < 1 or args.update_interval < 1 or args.alpha < 0: ap.error('seeds, ticks and update interval must be positive; alpha must be nonnegative')
    if not args.loads or any(rate < 0 for rate in args.loads): ap.error('loads must be nonnegative')
    os.makedirs(args.output_dir,exist_ok=True); rows=[]
    for rate in args.loads:
        for seed in range(args.seeds):
            for alg in ['baseline','queue_aware']: rows.append(Network(alg,seed,rate,args.ticks,args.alpha,args.update_interval,args.scenario).run())
    fields=list(rows[0]);
    with open(os.path.join(args.output_dir,'results.csv'),'w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    groups=defaultdict(list)
    for r in rows: groups[(r['traffic'],r['algorithm'])].append(r)
    summary=[]
    for (rate,alg), vals in groups.items():
        d={'traffic':rate,'algorithm':alg}
        for k in ['avg_delay','loss_ratio','throughput','route_changes','short_route_packets','middle_route_packets','lower_route_packets','cross_route_packets','pending_at_end']:
            d[k]=sum(x[k] for x in vals)/len(vals)
        summary.append(d)
    with open(os.path.join(args.output_dir,'summary.csv'),'w',newline='') as f: w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    plot(rows,args.output_dir)
    if args.table:
        lines=[f'Measured means | {args.seeds} matched seeds | {args.ticks} ticks/run',
               'Load  Algorithm     Delay(ticks)   Loss(%)   Throughput(pkt/tick)   Route changes',
               '----  ------------  ------------   -------   --------------------   -------------']
        for r in summary:
            lines.append(f"{r['traffic']:>4.1f}  {r['algorithm']:<12}  {r['avg_delay']:>12.3f}   "
                         f"{100*r['loss_ratio']:>7.3f}   {r['throughput']:>20.3f}   "
                         f"{r['route_changes']:>13.1f}")
        lines.append(f'Completed {len(rows)} runs; packet conservation passed in every run.')
    else:
        lines=[f'Wrote {len(rows)} runs to {args.output_dir}']+[str(r) for r in summary]
    for line in lines: print(line)
    if args.log: Path(args.log).write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
