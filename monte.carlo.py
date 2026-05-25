"""
DYPOD Track Profile Builder & Fleet Energy Simulator
=====================================================
• Parses DYPOD railML 3.2 (Správa železnic / OLTIS Group)
• Builds direction-aware stitched track profiles
• Interactive profile editor with via-waypoints
• Smart electrified-route search with detour analysis
• Physics simulation (Davis resistance, recuperation)
• Monte Carlo stochastic stop-probability analysis
"""
from __future__ import annotations
import glob, heapq, itertools, math, os, random, tempfile, time, zipfile
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── colour palette ────────────────────────────────────────────────────────────
C = dict(
    primary="#2563EB", secondary="#7C3AED", accent="#EA580C",
    green="#16A34A",   yellow="#CA8A04",    red="#DC2626",
    grey="#6B7280",    light="#F1F5F9",     dark="#1E293B",
    bg_blue="rgba(37,99,235,0.08)", bg_orange="rgba(234,88,12,0.08)",
    bg_green="rgba(22,163,74,0.10)",
)
ELEC_COLORS = {
    "NONE":"#9CA3AF", "25,000V/50Hz":"#16A34A",
    "3,000V/0Hz":"#7C3AED", "1,500V/0Hz":"#0891B2",
    "15,000V/16.7Hz":"#D97706",
}
# stop types: station=mandatory X, stoppingPoint=request R
PASSENGER_TYPES  = {"station", "stoppingPoint"}
MANDATORY_TYPES  = {"station"}
REQUEST_TYPES    = {"stoppingPoint"}

PREDEFINED_VEHICLES = {
    "EDITA (diesel railcar)":   dict(traction="DIESEL",  mass=22_000,  length=15,   power=152,   aux_power=20, accel=0.5, decel=0.8, efficiency=0.30),
    "EDITA+Btax":               dict(traction="DIESEL",  mass=42_000,  length=30,   power=152,   aux_power=25, accel=0.4, decel=0.8, efficiency=0.30),
    "RegioNova (Class 814)":    dict(traction="DIESEL",  mass=80_000,  length=44,   power=485,   aux_power=20, accel=0.5, decel=0.8, efficiency=0.32),
    "Stadler RS1 (Class 840)":  dict(traction="DIESEL",  mass=50_000,  length=25.5, power=514,   aux_power=25, accel=0.8, decel=0.9, efficiency=0.38),
    "CityElefant (Class 471)":  dict(traction="ELECTRIC",mass=155_000, length=79,   power=2_000, aux_power=80, accel=0.8, decel=0.9, efficiency=0.85),
}

# ─────────────────────────────────────────────────────────────────────────────
#  PARSER
# ─────────────────────────────────────────────────────────────────────────────
class DYPODParser:
    """Parse DYPOD railML 3.2 and expose graph + segment properties."""

    def __init__(self, xml_path: str):
        t0 = time.time()
        self._root = self._strip_ns(ET.parse(xml_path).getroot())
        self._esys:    dict[str,str]  = {}
        self._sub2seg: dict[str,str]  = {}
        self.op_info:  dict[str,dict] = {}
        self.seg_props:dict[str,dict] = {}
        self.graph:    dict[str,list] = {}
        self.op_by_name: dict[str,list] = {}
        self._parse_esys(); self._parse_sub2seg()
        self._parse_ops();  self._parse_segs(); self._build_graph()
        for oid,info in self.op_info.items():
            self.op_by_name.setdefault(info["name"],[]).append(oid)
        self.parse_time = round(time.time()-t0,2)

    @staticmethod
    def _strip_ns(root):
        for el in root.iter():
            if "}" in el.tag: el.tag=el.tag.split("}",1)[1]
        return root

    def _parse_esys(self):
        for es in self._root.iter("electrificationSystem"):
            v,f=es.get("voltage","0"),es.get("frequency","0")
            self._esys[es.get("id","")] = "NONE" if v=="0" else f"{int(float(v)):,}V/{f}Hz"

    def _parse_sub2seg(self):
        for ne in self._root.iter("netElement"):
            ecu=ne.find("elementCollectionUnordered")
            if ecu is not None:
                for ep in ecu.findall("elementPart"):
                    r=ep.get("ref","")
                    if r: self._sub2seg[r]=ne.get("id","")

    def _parse_ops(self):
        for op in self._root.iter("operationalPoint"):
            oid=op.get("id",""); nm=op.find("name")
            name=nm.get("name",oid) if nm is not None else oid
            lat=lon=None
            for sl in op.findall("spotLocation"):
                geo=sl.find("geometricCoordinate")
                if geo is not None and geo.get("positioningSystemRef","")=="gps01":
                    lat,lon=float(geo.get("x",0)),float(geo.get("y",0)); break
            types=[x.get("operationalType","") for x in op.iter("opOperation")]
            conn=[x.get("ref","") for x in op.findall("connectedToLine")]
            eq=op.find("opEquipment")
            n_tracks=int(eq.get("numberOfStationTracks",1)) if eq is not None else 1
            des=op.find("designator")
            sr70=des.get("entry","") if des is not None else ""
            self.op_info[oid]=dict(name=name,lat=lat,lon=lon,types=types,
                                   n_tracks=n_tracks,connected_lines=conn,sr70=sr70)

    def _parse_segs(self):
        speed_map:dict[str,dict]={}
        for ss in self._root.iter("speedSection"):
            ss_id=ss.get("id",""); maxsp=float(ss.get("maxSpeed",80))
            ane=ss.find(".//associatedNetElement")
            if ane is None: continue
            seg=self._sub2seg.get(ane.get("netElementRef",""),ane.get("netElementRef",""))
            d="reverse" if "_reverse" in ss_id else "normal"
            speed_map.setdefault(seg,{})[d]=maxsp

        grad_map:dict[str,dict]={}
        for gc in self._root.iter("gradientCurve"):
            grad=float(gc.get("gradient",0)); loc=gc.find("linearLocation")
            if loc is None: continue
            ane=loc.find("associatedNetElement")
            if ane is None: continue
            grad_map.setdefault(ane.get("netElementRef",""),{})[loc.get("applicationDirection","normal")]=grad

        elec_map:dict[str,str]={}
        for sec in self._root.iter("electrificationSection"):
            ane=sec.find(".//associatedNetElement"); esr=sec.find("electrificationSystemRef")
            if ane is None or esr is None: continue
            seg=self._sub2seg.get(ane.get("netElementRef",""),ane.get("netElementRef",""))
            label=self._esys.get(esr.get("ref",""),"UNKNOWN")
            elec_map[seg]=label if label!="NONE" else elec_map.get(seg,"NONE")

        gps_map:dict[str,tuple]={}
        for ne in self._root.iter("netElement"):
            ne_id=ne.get("id","")
            for aps in ne.findall("associatedPositioningSystem"):
                if aps.get("positioningSystemRef","")!="gps01": continue
                ics=aps.findall("intrinsicCoordinate")
                if len(ics)<2: continue
                coords=[]
                for ic in ics:
                    geo=ic.find("geometricCoordinate")
                    if geo is not None: coords.append((float(geo.get("x",0)),float(geo.get("y",0))))
                if len(coords)>=2: gps_map[ne_id]=(coords[0],coords[-1]); break

        len_map:dict[str,float]={}
        for tr in self._root.iter("track"):
            ane=tr.find(".//associatedNetElement"); lel=tr.find("length")
            if ane is None or lel is None: continue
            seg=self._sub2seg.get(ane.get("netElementRef",""),ane.get("netElementRef",""))
            ltype,val=lel.get("type",""),float(lel.get("value",0))
            if ltype=="physical" or seg not in len_map: len_map[seg]=val

        for line in self._root.iter("line"):
            ane=line.find(".//associatedNetElement")
            if ane is None: continue
            ne_ref=ane.get("netElementRef","")
            lel=line.find("length")
            length_m=float(lel.get("value",0)) if lel is not None else len_map.get(ne_ref,0.0)
            sp=speed_map.get(ne_ref,{}); gr=grad_map.get(ne_ref,{})
            electr=elec_map.get(ne_ref,"NONE"); gps=gps_map.get(ne_ref,(None,None))
            lp=line.find("linePerformance"); ll=line.find("lineLayout"); lo=line.find("lineOperation")
            self.seg_props[ne_ref]=dict(
                line_id=line.get("id",""), length_m=length_m,
                speed_normal=sp.get("normal",sp.get("reverse",80.0)),
                speed_reverse=sp.get("reverse",sp.get("normal",80.0)),
                grad_normal=gr.get("normal",0.0), grad_reverse=gr.get("reverse",0.0),
                electrification=electr, recuperation=0 if electr=="NONE" else 1,
                braking_dist=int(lp.get("signalledBrakingDistance",0)) if lp is not None else 0,
                n_tracks=(ll.get("numberOfTracks","single") if ll is not None else "single"),
                mode_of_op=(lo.get("modeOfOperation","") if lo is not None else ""),
                gps_start=gps[0], gps_end=gps[1],
            )

    def _build_graph(self):
        for line in self._root.iter("line"):
            b_el=line.find("beginsInOP"); e_el=line.find("endsInOP")
            ane=line.find(".//associatedNetElement")
            if b_el is None or e_el is None or ane is None: continue
            b_op,e_op=b_el.get("ref",""),e_el.get("ref","")
            ne_ref=ane.get("netElementRef","")
            props=self.seg_props.get(ne_ref,{})
            length_m=props.get("length_m",0.0); electr=props.get("electrification","NONE")
            self.graph.setdefault(b_op,[]).append(dict(to=e_op,ne_id=ne_ref,length_m=length_m,forward=True,electr=electr))
            self.graph.setdefault(e_op,[]).append(dict(to=b_op,ne_id=ne_ref,length_m=length_m,forward=False,electr=electr))

    # ── path search ──────────────────────────────────────────────────────────
    def dijkstra(self, start:str, end:str, unelec_penalty_m:float=0.0):
        if start not in self.graph: return None,[]
        tb=itertools.count(); pq=[(0.0,next(tb),start,[])]
        vis:set=set()
        while pq:
            cost,_,node,path=heapq.heappop(pq)
            if node in vis: continue
            vis.add(node)
            if node==end: return cost,path
            for edge in self.graph.get(node,[]):
                nxt=edge["to"]
                if nxt in vis: continue
                pen=unelec_penalty_m if edge["electr"]=="NONE" else 0.0
                heapq.heappush(pq,(cost+edge["length_m"]+pen,next(tb),nxt,
                                   path+[dict(from_op=node,**edge)]))
        return None,[]

    # ── electrified route analysis ────────────────────────────────────────────
    def analyse_electrification(self, start_op:str, end_op:str) -> dict:
        """
        Returns a dict describing the electrification situation:
          normal_km        total km of normal shortest path
          normal_ue_km     unelectrified km on normal path
          gateway_km       unavoidable unelec run-in from start (km)
          gateway_op       first electrified op reachable from start
          penalised_km     km of penalised (electrified-preferring) path
          penalised_ue_km  unelec km on penalised path
          detour_km        extra km of penalised vs normal
          elec_saving_km   unelec km saved by penalised route
          has_alternative  whether penalised route meaningfully differs
        """
        _,p_normal = self.dijkstra(start_op, end_op, 0)
        if not p_normal:
            return dict(normal_km=0,normal_ue_km=0,gateway_km=0,gateway_op=None,
                        penalised_km=0,penalised_ue_km=0,detour_km=0,
                        elec_saving_km=0,has_alternative=False)

        def ue_km(path): return sum(self.seg_props.get(s["ne_id"],{}).get("length_m",0)
                                    for s in path if s["electr"]=="NONE")/1000
        def tot_km(path): return sum(self.seg_props.get(s["ne_id"],{}).get("length_m",0)
                                     for s in path)/1000

        normal_ue = ue_km(p_normal)
        normal_tot= tot_km(p_normal)

        # Find unavoidable gateway (how far until first electrified segment)
        gateway_km=0.0; gateway_op=None
        for s in p_normal:
            if s["electr"]!="NONE":
                gateway_op=s["from_op"]; break
            gateway_km+=self.seg_props.get(s["ne_id"],{}).get("length_m",0)/1000

        # Penalised path (100km penalty per NONE segment)
        _,p_pen = self.dijkstra(start_op, end_op, 100_000)
        pen_ue  = ue_km(p_pen)
        pen_tot = tot_km(p_pen)

        elec_saving = normal_ue - pen_ue
        detour      = pen_tot  - normal_tot
        # Useful if we save ≥1km unelec AND detour ≤ 5×saving
        has_alt = (elec_saving >= 1.0) and (detour <= max(elec_saving*10, 30))

        return dict(
            normal_km=round(normal_tot,1), normal_ue_km=round(normal_ue,1),
            gateway_km=round(gateway_km,1), gateway_op=gateway_op,
            penalised_km=round(pen_tot,1),  penalised_ue_km=round(pen_ue,1),
            detour_km=round(detour,1),      elec_saving_km=round(elec_saving,1),
            has_alternative=has_alt,
        )

    # ── profile builder ───────────────────────────────────────────────────────
    def build_profile(self, start_op:str, end_op:str,
                      via_ops:list[str]|None=None,
                      unelec_penalty_m:float=0.0) -> pd.DataFrame:
        waypoints=[start_op]+(via_ops or [])+[end_op]
        all_steps:list[dict]=[]
        for i in range(len(waypoints)-1):
            _,path=self.dijkstra(waypoints[i],waypoints[i+1],unelec_penalty_m)
            if not path: return pd.DataFrame()
            all_steps.extend(path)
        if not all_steps: return pd.DataFrame()

        def _stop_type(oid:str)->str:
            types=self.op_info.get(oid,{}).get("types",[])
            if any(t in MANDATORY_TYPES for t in types): return "X"
            if any(t in REQUEST_TYPES  for t in types): return "R"
            return ""

        def _sd(seg:dict,fwd:bool)->dict:
            return dict(
                speed_kmh      =seg.get("speed_normal" if fwd else "speed_reverse",80.0),
                gradient_perm  =seg.get("grad_normal"  if fwd else "grad_reverse", 0.0),
                electrification=seg.get("electrification","NONE"),
                recuperation   =seg.get("recuperation",0),
                length_m       =seg.get("length_m",0.0),
                braking_dist   =seg.get("braking_dist",0),
                n_tracks       =seg.get("n_tracks","single"),
            )

        rows:list[dict]=[]; cum_m=0.0
        for step in all_steps:
            from_op=step["from_op"]; seg=self.seg_props.get(step["ne_id"],{})
            sd=_sd(seg,step["forward"]); info=self.op_info.get(from_op,{})
            rows.append(dict(cum_km=cum_m/1000.0, op_id=from_op,
                             station_name=info.get("name",from_op),
                             stop_type=_stop_type(from_op),
                             lat=info.get("lat"), lon=info.get("lon"),
                             op_types=", ".join(info.get("types",[])), **sd))
            cum_m+=seg.get("length_m",0.0)

        last_fwd=all_steps[-1]["forward"]; last_seg=self.seg_props.get(all_steps[-1]["ne_id"],{})
        last_sd=_sd(last_seg,last_fwd); info_end=self.op_info.get(end_op,{})
        rows.append(dict(cum_km=cum_m/1000.0, op_id=end_op,
                         station_name=info_end.get("name",end_op), stop_type="X",
                         lat=info_end.get("lat"), lon=info_end.get("lon"),
                         op_types=", ".join(info_end.get("types",[])), length_m=0.0,
                         **{k:v for k,v in last_sd.items() if k!="length_m"}))
        df=pd.DataFrame(rows); df["cum_km"]=df["cum_km"].round(4)
        return df

    def search_stations(self,query:str,max_results=60)->list[tuple[str,str]]:
        q=query.strip().lower()
        if not q: return []
        out=[(oid,info["name"]) for oid,info in self.op_info.items()
             if any(t in PASSENGER_TYPES for t in info.get("types",[]))
             and q in info["name"].lower()]
        return sorted(out,key=lambda x:x[1])[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK PROFILE
# ─────────────────────────────────────────────────────────────────────────────
class TrackProfile:
    def __init__(self,df:pd.DataFrame):
        self.df=df.copy().reset_index(drop=True)
        self.segments=self._build_segs(); self.stations=self._build_stations()
        self.total_km=float(df["cum_km"].max())

    def _build_segs(self):
        segs=[]
        for i in range(len(self.df)-1):
            r=self.df.iloc[i]
            segs.append(dict(km_start=float(r["cum_km"]),km_end=float(self.df.iloc[i+1]["cum_km"]),
                             v_limit=float(r["speed_kmh"])/3.6, grad=float(r["gradient_perm"])/1000.0,
                             electrification=str(r.get("electrification","NONE")),
                             recuperation=int(r.get("recuperation",0))))
        return segs

    def _build_stations(self):
        return [dict(name=str(r.get("station_name","")),km=float(r["cum_km"]),
                     type=str(r.get("stop_type","")).upper())
                for _,r in self.df.iterrows()
                if str(r.get("stop_type","")).upper() in ("X","R")
                and str(r.get("station_name","")).strip()]

    def seg_at(self,km:float)->dict:
        for s in self.segments:
            if s["km_start"]<=km<=s["km_end"]+1e-6: return s
        return self.segments[-1] if self.segments else {}

    def v_limit_span(self,front_km:float,rear_km:float)->float:
        lo,hi=min(front_km,rear_km),max(front_km,rear_km)
        lims=[s["v_limit"] for s in self.segments if lo<=s["km_end"]+1e-6 and hi>=s["km_start"]-1e-6]
        return min(lims) if lims else 0.0

    @property
    def n_mandatory(self)->int:
        return sum(1 for s in self.stations if s["type"]=="X")

    @property
    def n_request(self)->int:
        return sum(1 for s in self.stations if s["type"]=="R")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class TrainSimulator:
    def __init__(self,mass_kg,length_m,max_power_kw,aux_power_kw,
                 max_accel,max_decel,traction_type,efficiency):
        self.mass_kg=mass_kg; self.eff_mass=mass_kg*1.08
        self.A,self.B,self.C=1500.0,30.0,4.0
        self.traction=traction_type
        self.trac_eff=efficiency if traction_type=="ELECTRIC" else 0.85
        self.regen_eff=0.75 if traction_type=="ELECTRIC" else 0.0
        self.aux_w=aux_power_kw*1_000.0; self.max_w=max_power_kw*1_000.0
        self.max_accel=max_accel; self.max_decel=max_decel; self.length_m=length_m

    def _res(self,v): return self.A+self.B*v+self.C*v*v

    def run(self,track:TrackProfile,stop_mode="all",stop_prob=1.0,
            dwell=30.0,record=False):
        total_m=track.total_km*1000.0; g=9.8186
        stops_km,stop_names=[],[]
        for st in track.stations:
            km=st["km"]
            if km<=1e-3 or km>=track.total_km-1e-3: continue
            will=(st["type"]=="X") or \
                 (st["type"]=="R" and ((stop_mode=="all") or
                  (stop_mode=="random" and random.random()<=stop_prob)))
            if will: stops_km.append(km); stop_names.append(st["name"])

        v=dist=e_j=r_j=t_s=0.0
        hist={k:[] for k in ("time_s","km","v_kmh","v_limit_kmh",
                              "gross_kwh","regen_kwh","net_kwh")} if record else None

        while dist<total_m:
            km=dist/1000.0; rear_km=max(0.0,km-self.length_m/1000.0)
            seg=track.seg_at(km); v_lim=track.v_limit_span(km,rear_km)
            slope=seg.get("grad",0.0); electr=seg.get("electrification","NONE")
            recup=seg.get("recuperation",0)==1

            if self.traction=="ELECTRIC" and electr=="NONE":
                raise RuntimeError(
                    f"⚡ Electric vehicle on non-electrified track at km {km:.2f}. "
                    "Enable 'Prefer electrified route' or choose a diesel vehicle.")

            f_grad=self.mass_kg*g*slope
            eff_decel=max(0.05,self.max_decel+g*slope)
            max_safe=v_lim
            for s_km in stops_km:
                if s_km<km-0.01: continue
                d2s=(s_km-km)*1000.0
                max_safe=min(max_safe,math.sqrt(max(0.0,2.0*eff_decel*d2s)))

            if hist:
                hist["time_s"].append(t_s); hist["km"].append(km)
                hist["v_kmh"].append(v*3.6); hist["v_limit_kmh"].append(v_lim*3.6)
                hist["gross_kwh"].append(e_j/3_600_000.0)
                hist["regen_kwh"].append(r_j/3_600_000.0)
                hist["net_kwh"].append((e_j-r_j)/3_600_000.0)

            v_eff=max(v,0.5); mech_p=reg_p=0.0
            if v>max_safe+1e-4:
                f_res=self._res(v); nat_d=(f_res+f_grad)/self.eff_mass
                brk=max(0.0,min(self.max_decel,(v-max_safe)/1.0-nat_d))
                if recup: reg_p=self.eff_mass*brk*v*self.regen_eff
                v=max(0.0,v-(brk+nat_d)*1.0)
            elif v<min(v_lim,max_safe)-1e-4:
                f_res=self._res(v)
                des_f=f_res+self.eff_mass*self.max_accel+f_grad
                act_f=max(0.0,min(des_f,self.max_w/v_eff))
                v=max(0.0,min(v+(act_f-f_res-f_grad)/self.eff_mass,v_lim,max_safe))
                mech_p=max(0.0,act_f*v)
            else:
                f_res=self._res(v); act_f=max(0.0,min(f_res+f_grad,self.max_w/v_eff))
                v=max(0.0,v+(act_f-f_res-f_grad)/self.eff_mass); mech_p=max(0.0,act_f*v)

            e_j+=(mech_p/self.trac_eff+self.aux_w); r_j+=reg_p; dist+=v; t_s+=1.0

            if v<0.5 and any(abs(km-s)<=0.05 for s in stops_km):
                v=0.0
                if hist:
                    for pn in range(2):
                        vl=track.v_limit_span(km,km)*3.6
                        hist["time_s"].append(t_s); hist["km"].append(km)
                        hist["v_kmh"].append(0.0); hist["v_limit_kmh"].append(vl)
                        hist["gross_kwh"].append(e_j/3_600_000.0)
                        hist["regen_kwh"].append(r_j/3_600_000.0)
                        hist["net_kwh"].append((e_j-r_j)/3_600_000.0)
                        if pn==0: e_j+=self.aux_w*dwell; t_s+=dwell
                else:
                    e_j+=self.aux_w*dwell; t_s+=dwell
                stops_km=[s for s in stops_km if abs(s-km)>0.05]

        return hist,stop_names,dict(gross_kwh=e_j/3_600_000.0,regen_kwh=r_j/3_600_000.0,
                                    net_kwh=(e_j-r_j)/3_600_000.0,journey_time_s=t_s)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_dur(s):
    s=int(s); h,r=divmod(s,3600); m,sc=divmod(r,60)
    return f"{h:02d}h {m:02d}m {sc:02d}s" if h else f"{m:02d}m {sc:02d}s"

def to_unit(stats,traction,density,eff):
    return stats["net_kwh"]/(density*eff) if traction=="DIESEL" else stats["net_kwh"]

def elec_color(label:str)->str:
    for k,v in ELEC_COLORS.items():
        if k in label: return v
    return ELEC_COLORS["NONE"]

def kpi_card(val:str,lbl:str,delta:str="",color:str=C["primary"])->str:
    d=f'<div style="font-size:.75rem;color:{color};margin-top:3px">{delta}</div>' if delta else ""
    return (f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
            f'padding:14px 18px;text-align:center;height:90px;display:flex;flex-direction:column;'
            f'justify-content:center">'
            f'<div style="font-size:1.45rem;font-weight:700;color:#1E3A5F;line-height:1.2">{val}</div>'
            f'<div style="font-size:.75rem;color:#64748B;margin-top:3px">{lbl}</div>{d}</div>')

def load_xml_from_upload(uploaded)->str|None:
    ext=uploaded.name.lower().rsplit(".",1)[-1]
    tmp=tempfile.mkdtemp(prefix="dypod_")
    if ext=="zip":
        zp=os.path.join(tmp,uploaded.name)
        with open(zp,"wb") as f: f.write(uploaded.read())
        with zipfile.ZipFile(zp) as zf:
            xmls=[n for n in zf.namelist() if n.lower().endswith((".xml",".railml"))]
            if not xmls: return None
            zf.extract(xmls[0],tmp); return os.path.join(tmp,xmls[0])
    path=os.path.join(tmp,uploaded.name)
    with open(path,"wb") as f: f.write(uploaded.read())
    return path

def make_profile_chart(df:pd.DataFrame)->go.Figure:
    """Shared 3-panel chart: speed / gradient / electrification."""
    total_km=float(df["cum_km"].max()); stops=df[df["stop_type"].isin(["X","R"])]
    bar_w=max(total_km/max(len(df),1)*0.85,0.01)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      subplot_titles=("Speed profile [km/h]",
                                      "Gradient [‰]  (↑ uphill  ↓ downhill)",
                                      "Electrification"),
                      row_heights=[0.48,0.32,0.20],vertical_spacing=0.055)

    fig.add_trace(go.Scatter(x=df["cum_km"],y=df["speed_kmh"],mode="lines",
        name="Speed limit",line=dict(color=C["primary"],width=2.5),
        fill="tozeroy",fillcolor=C["bg_blue"]),row=1,col=1)

    for stype,sym,col,nm in [("X","diamond",C["accent"],"Station (X)"),
                              ("R","circle",C["yellow"],"Halt (R)")]:
        sg=stops[stops["stop_type"]==stype]
        if sg.empty: continue
        fig.add_trace(go.Scatter(x=sg["cum_km"],y=sg["speed_kmh"],mode="markers+text",
            marker=dict(symbol=sym,size=9,color=col,line=dict(color="white",width=1.5)),
            text=sg["station_name"],textposition="top center",textfont=dict(size=7),
            name=nm,hovertemplate="<b>%{text}</b><br>km %{x:.2f}<br>%{y:.0f} km/h<extra></extra>"),
            row=1,col=1)

    fig.add_trace(go.Bar(x=df["cum_km"],y=df["gradient_perm"].clip(lower=0),
        name="Uphill",marker_color=C["red"],width=bar_w,
        hovertemplate="km %{x:.2f}<br>+%{y:.1f} ‰<extra></extra>"),row=2,col=1)
    fig.add_trace(go.Bar(x=df["cum_km"],y=df["gradient_perm"].clip(upper=0),
        name="Downhill",marker_color=C["primary"],width=bar_w,
        hovertemplate="km %{x:.2f}<br>%{y:.1f} ‰<extra></extra>"),row=2,col=1)

    fig.add_trace(go.Bar(x=df["cum_km"],y=[1]*len(df),
        marker_color=[elec_color(e) for e in df["electrification"]],
        name="Electrification",hovertext=df["electrification"],
        hovertemplate="%{hovertext}<br>km %{x:.2f}<extra></extra>",
        width=bar_w,showlegend=False),row=3,col=1)

    for _,sr in stops[stops["stop_type"]=="X"].iterrows():
        fig.add_vline(x=sr["cum_km"],line_width=0.7,line_dash="dot",
                      line_color="#CBD5E1",row=1,col=1)

    fig.update_xaxes(title_text="Distance from departure [km]",row=3,col=1,gridcolor=C["light"])
    fig.update_yaxes(title_text="Speed [km/h]", row=1,col=1,gridcolor=C["light"])
    fig.update_yaxes(title_text="Gradient [‰]", row=2,col=1,gridcolor=C["light"],
                     zeroline=True,zerolinecolor="#CBD5E1")
    fig.update_yaxes(showticklabels=False,row=3,col=1)
    fig.update_layout(height=720,barmode="overlay",paper_bgcolor="white",plot_bgcolor="white",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,
                    bgcolor="rgba(255,255,255,0.9)",bordercolor="#E2E8F0",borderwidth=1),
        margin=dict(t=70,b=10,l=60,r=20),font=dict(family="Inter,sans-serif",size=12))
    fig.update_xaxes(gridcolor=C["light"],showgrid=True)
    return fig

def make_route_map(df:pd.DataFrame, via_ops:list, parser:DYPODParser,
                   show_all_stops:bool=True)->go.Figure:
    map_df=df.dropna(subset=["lat","lon"])
    fig=go.Figure()
    for sys in df["electrification"].unique():
        seg=map_df[map_df["electrification"]==sys]
        if seg.empty: continue
        fig.add_trace(go.Scattermapbox(lat=seg["lat"],lon=seg["lon"],mode="lines",
            line=dict(width=5,color=elec_color(sys)),name=sys,
            hovertemplate=f"{sys}<extra></extra>"))

    if show_all_stops:
        all_pts=map_df[map_df["stop_type"].isin(["X","R"])]
        r_pts=all_pts[all_pts["stop_type"]=="R"]
        if not r_pts.empty:
            fig.add_trace(go.Scattermapbox(lat=r_pts["lat"],lon=r_pts["lon"],mode="markers",
                marker=dict(size=7,color=C["yellow"]),name="Halt (R)",
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                customdata=r_pts["station_name"]))

    x_pts=map_df[map_df["stop_type"]=="X"]
    if not x_pts.empty:
        fig.add_trace(go.Scattermapbox(lat=x_pts["lat"],lon=x_pts["lon"],
            mode="markers+text",marker=dict(size=11,color=C["accent"]),
            text=x_pts["station_name"],textposition="top right",
            textfont=dict(size=9,color=C["dark"]),name="Station (X)",
            hovertemplate="<b>%{text}</b><br>km %{customdata[0]:.2f}<extra></extra>",
            customdata=x_pts[["cum_km"]].values))

    for vid in (via_ops or []):
        info=parser.op_info.get(vid,{})
        if info.get("lat") and info.get("lon"):
            fig.add_trace(go.Scattermapbox(lat=[info["lat"]],lon=[info["lon"]],
                mode="markers+text",marker=dict(size=18,color=C["yellow"],symbol="circle"),
                text=[info["name"]],textposition="top right",
                textfont=dict(size=10,color="#92400E"),name=f"Via: {info['name']}",showlegend=False))

    if not map_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=[map_df["lat"].iloc[0],map_df["lat"].iloc[-1]],
            lon=[map_df["lon"].iloc[0],map_df["lon"].iloc[-1]],
            mode="markers+text",
            marker=dict(size=20,color=[C["green"],C["red"]]),
            text=[map_df["station_name"].iloc[0],map_df["station_name"].iloc[-1]],
            textposition=["top right","top left"],
            textfont=dict(size=11,color=C["dark"],family="Inter,sans-serif"),
            name="Start / End",hovertemplate="<b>%{text}</b><extra></extra>"))

    lat_c=float(map_df["lat"].mean()) if not map_df.empty else 50.0
    lon_c=float(map_df["lon"].mean()) if not map_df.empty else 15.0
    fig.update_layout(mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=lat_c,lon=lon_c),zoom=7),
        height=520,margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.92)",bordercolor="#E2E8F0",
                    borderwidth=1,x=0.01,y=0.99,font=dict(size=11)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DYPOD Simulator",page_icon="🚆",layout="wide")
st.markdown("""<style>
  [data-testid="stSidebar"]{background:#F1F5F9}
  .stButton>button{border-radius:8px;font-weight:600;transition:all .15s}
  .stButton>button:hover{transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,.15)}
  .block-container{padding-top:1.5rem}
  .stTabs [data-baseweb="tab"]{font-weight:600;font-size:.92rem}
  .stTabs [data-baseweb="tab-list"]{gap:4px}
  h3{color:#1E3A5F;font-size:1.2rem}
  .section-hdr{font-size:.95rem;font-weight:700;color:#1E3A5F;
               border-left:3px solid #2563EB;padding-left:8px;margin:14px 0 6px}
  .info-box{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
            padding:10px 14px;font-size:.85rem;color:#1E40AF;margin:8px 0}
  .warn-box{background:#FEF9C3;border:1px solid #FDE047;border-radius:8px;
            padding:10px 14px;font-size:.85rem;color:#713F12;margin:8px 0}
  .danger-box{background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;
              padding:10px 14px;font-size:.85rem;color:#7F1D1D;margin:8px 0}
  .ok-box{background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;
          padding:10px 14px;font-size:.85rem;color:#14532D;margin:8px 0}
</style>""",unsafe_allow_html=True)

for _k in ("parser","xml_path","profile_df","via_ops","rep_result","mc_result",
           "elec_analysis","op_start","op_end"):
    if _k not in st.session_state: st.session_state[_k]=None
if not isinstance(st.session_state.via_ops,list): st.session_state.via_ops=[]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚆 DYPOD Simulator")
    st.caption("Czech railway track profile & energy analysis")
    st.markdown("---")

    # 1. File
    st.markdown('<div class="section-hdr">📂 Load railML</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload railML or zip",type=["xml","railml","zip"],
                               label_visibility="collapsed")
    xml_path=None
    if uploaded:
        xp=load_xml_from_upload(uploaded)
        if xp and xp!=st.session_state.xml_path:
            st.session_state.update(xml_path=xp,parser=None,profile_df=None,
                                    via_ops=[],rep_result=None,mc_result=None,
                                    elec_analysis=None)
        xml_path=st.session_state.xml_path
    else:
        local=sorted(glob.glob("*.xml")+glob.glob("*.railml")+glob.glob("*.zip")+
                     glob.glob("/tmp/*.xml")+glob.glob("/tmp/*.railml"))
        if local:
            sel=st.selectbox("Or pick local file",local,label_visibility="collapsed")
            if sel!=st.session_state.xml_path:
                st.session_state.update(xml_path=sel,parser=None,profile_df=None,
                                        via_ops=[],rep_result=None,mc_result=None,
                                        elec_analysis=None)
            xml_path=sel

    @st.cache_resource(show_spinner="Parsing railML infrastructure…")
    def _load(path:str)->DYPODParser:
        if path.lower().endswith(".zip"):
            tmp=tempfile.mkdtemp()
            with zipfile.ZipFile(path) as zf:
                xmls=[n for n in zf.namelist() if n.lower().endswith((".xml",".railml"))]
                zf.extract(xmls[0],tmp); path=os.path.join(tmp,xmls[0])
        return DYPODParser(path)

    parser=None
    if xml_path:
        try:
            parser=_load(xml_path)
            st.success(f"✅ {parser.parse_time}s · **{len(parser.op_info):,}** OPs · **{len(parser.seg_props):,}** segs")
        except Exception as exc:
            st.error(f"Parse error: {exc}"); st.stop()
    else:
        st.info("Upload a DYPOD railML file to begin."); st.stop()

    # 2. Route
    st.markdown('<div class="section-hdr">🗺️ Route</div>',unsafe_allow_html=True)
    qa=st.text_input("Departure station","",placeholder="e.g. Praha hl.n.",key="q_dep")
    qb=st.text_input("Arrival station","",placeholder="e.g. Brno hl.n.",key="q_arr")
    res_a=parser.search_stations(qa) if qa.strip() else []
    res_b=parser.search_stations(qb) if qb.strip() else []
    op_start=op_end=None
    if res_a:
        la=st.selectbox("↳ Departure",options=[r[1] for r in res_a],key="sel_dep",label_visibility="collapsed")
        op_start=next((r[0] for r in res_a if r[1]==la),None)
    elif qa.strip(): st.caption("⚠️ No station found")
    if res_b:
        lb=st.selectbox("↳ Arrival",options=[r[1] for r in res_b],key="sel_arr",label_visibility="collapsed")
        op_end=next((r[0] for r in res_b if r[1]==lb),None)
    elif qb.strip(): st.caption("⚠️ No station found")

    st.markdown("**⚡ Electrified route preference**")
    elec_reroute=st.toggle("Prefer electrified track",value=False,
                            help="Adds a penalty to non-electrified segments. "
                                 "Useful for electric vehicles.")
    pen_km=100
    if elec_reroute:
        pen_km=st.slider("Detour tolerance (km equivalent)",10,500,100,10,
                          help="How many km of detour are acceptable to avoid 1km of non-electrified track.")

    btn_profile=st.button("🗺️ Build Track Profile",use_container_width=True,type="primary",
                           disabled=not(op_start and op_end and op_start!=op_end))

    # 3. Vehicle
    st.markdown('<div class="section-hdr">🚃 Vehicle</div>',unsafe_allow_html=True)
    veh=st.selectbox("Preset",["Custom"]+list(PREDEFINED_VEHICLES.keys()),label_visibility="collapsed")
    if veh=="Custom":
        traction=st.selectbox("Traction",["DIESEL","ELECTRIC"])
        col_v1,col_v2=st.columns(2)
        mass=col_v1.number_input("Mass (kg)",value=100_000,step=5_000)
        length=col_v2.number_input("Length (m)",value=40,step=5)
        power=col_v1.number_input("Power (kW)",value=500,step=50)
        aux_power=col_v2.number_input("Aux (kW)",value=40,step=5)
        accel=st.slider("Accel (m/s²)",0.2,1.5,0.6,0.05)
        decel=st.slider("Brake (m/s²)",0.4,1.5,0.9,0.05)
        efficiency=st.slider("Efficiency (%)",15,95,35)/100.0
        diesel_d=st.number_input("Diesel density (kWh/L)",value=10.0,step=0.1)
    else:
        p=PREDEFINED_VEHICLES[veh]
        traction,mass,length=p["traction"],p["mass"],p["length"]
        power,aux_power=p["power"],p["aux_power"]
        accel,decel,efficiency=p["accel"],p["decel"],p["efficiency"]
        diesel_d=10.0
    unit_lbl="L fuel" if traction=="DIESEL" else "kWh"

    # 4. Simulation
    st.markdown('<div class="section-hdr">⚙️ Simulation</div>',unsafe_allow_html=True)
    dwell=st.number_input("Station dwell (s)",value=30,step=5)
    stop_mode=st.segmented_control("Stop mode",
        options=["all","random","none"],
        format_func={"all":"All stops","random":"Stochastic","none":"Express"}.get,
        default="all")
    stop_prob=1.0
    if stop_mode=="random":
        stop_prob=st.slider("Request-stop probability",0.0,1.0,0.5,0.05)

    st.markdown('<div class="section-hdr">🎲 Monte Carlo</div>',unsafe_allow_html=True)
    mc_n=st.number_input("Runs per probability",20,500,100,10)
    mc_probs=st.multiselect("Probabilities to sweep",
                             options=[1.0,0.8,0.6,0.4,0.2,0.0],
                             default=[1.0,0.8,0.6,0.4,0.2,0.0])

    c1,c2=st.columns(2)
    btn_run=c1.button("▶️ Run",  use_container_width=True,
                       disabled=st.session_state.profile_df is None)
    btn_mc =c2.button("🎲 MC",   use_container_width=True,
                       disabled=st.session_state.profile_df is None)

# ─── Actions ─────────────────────────────────────────────────────────────────
if btn_profile and op_start and op_end:
    pen=pen_km*1000.0 if elec_reroute else 0.0
    with st.spinner("Finding shortest path and stitching profile…"):
        ea=parser.analyse_electrification(op_start,op_end)
        df=parser.build_profile(op_start,op_end,
                                 via_ops=st.session_state.via_ops or [],
                                 unelec_penalty_m=pen)
    if df.empty:
        st.error("No path found between the selected stations.")
    else:
        st.session_state.update(profile_df=df,op_start=op_start,op_end=op_end,
                                 elec_analysis=ea,rep_result=None,mc_result=None)

if btn_run and st.session_state.profile_df is not None:
    with st.spinner("Running physics simulation…"):
        try:
            track=TrackProfile(st.session_state.profile_df)
            sim=TrainSimulator(mass,length,power,aux_power,accel,decel,traction,efficiency)
            hist,snames,stats=sim.run(track,stop_mode=stop_mode,stop_prob=stop_prob,
                                      dwell=dwell,record=True)
            st.session_state.rep_result=dict(hist=hist,stop_names=snames,stats=stats,
                traction=traction,diesel_d=diesel_d,efficiency=efficiency,
                unit_lbl=unit_lbl,vehicle_name=veh,dwell=dwell)
        except RuntimeError as e:
            st.error(str(e))

if btn_mc and st.session_state.profile_df is not None and mc_probs:
    with st.spinner(f"Monte Carlo — {int(mc_n)} × {len(mc_probs)} runs…"):
        try:
            track=TrackProfile(st.session_state.profile_df)
            sim=TrainSimulator(mass,length,power,aux_power,accel,decel,traction,efficiency)
            rows_mc=[]
            for p_val in sorted(mc_probs,reverse=True):
                sm="all" if p_val==1.0 else "none" if p_val==0.0 else "random"
                e_list,t_list=[],[]
                for _ in range(int(mc_n)):
                    _,_,st_=sim.run(track,sm,p_val,dwell)
                    e_list.append(to_unit(st_,traction,diesel_d,efficiency))
                    t_list.append(st_["journey_time_s"])
                rows_mc.append(dict(prob=f"{int(p_val*100)}%",p_num=p_val,
                    mean_e=np.mean(e_list),std_e=np.std(e_list),
                    min_e=np.min(e_list),max_e=np.max(e_list),
                    mean_t=np.mean(t_list),min_t=np.min(t_list),max_t=np.max(t_list)))
            mc_df=pd.DataFrame(rows_mc)
            mc_df["savings"]=(mc_df["mean_e"].max()-mc_df["mean_e"]).clip(lower=0)
            mc_df["unit"]=unit_lbl
            st.session_state.mc_result=mc_df
        except RuntimeError as e:
            st.error(str(e))

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_prof,tab_edit,tab_run_t,tab_mc_t=st.tabs([
    "🗺️  Track Profile","✏️  Profile Editor","▶️  Run","🎲  Monte Carlo"])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – TRACK PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prof:
    df=st.session_state.profile_df
    ea=st.session_state.elec_analysis
    if df is None:
        st.markdown("### 👈 Search for stations in the sidebar and click **Build Track Profile**")
        st.info("The tool stitches meso-level railML segments into a direction-aware profile "
                "including speed limits, gradient, electrification, and GPS coordinates.")
        st.stop()

    total_km=float(df["cum_km"].max())
    stops_x=df[df["stop_type"]=="X"]; stops_r=df[df["stop_type"]=="R"]
    ue_km=df[df["electrification"]=="NONE"]["length_m"].sum()/1000
    elec_km=total_km-ue_km
    max_spd=df["speed_kmh"].max(); max_grad=df["gradient_perm"].abs().max()
    sn=df["station_name"].iloc[0]; en=df["station_name"].iloc[-1]

    st.markdown(f"### 📍 {sn}  →  {en}")

    # Electrification alerts
    if ea and traction=="ELECTRIC":
        ue=ea["normal_ue_km"]
        gw=ea["gateway_km"]
        if ue==0:
            st.markdown('<div class="ok-box">✅ <b>Fully electrified route</b> — electric vehicle can run without restriction.</div>',unsafe_allow_html=True)
        elif gw>0 and gw==ue:
            st.markdown(f'<div class="warn-box">⚠️ <b>Unavoidable non-electrified run-in: {gw:.1f} km</b> from the departure station to the electrified network. '
                        f'No alternative electrified route exists out of <b>{sn}</b>. '
                        f'Consider starting from the nearest electrified station instead.</div>',unsafe_allow_html=True)
        elif ea.get("has_alternative") and not elec_reroute:
            st.markdown(f'<div class="warn-box">⚡ <b>{ue:.1f} km of non-electrified track</b> on this route '
                        f'(+{gw:.1f} km unavoidable run-in). '
                        f'Enable <b>Prefer electrified route</b> in the sidebar to save <b>{ea["elec_saving_km"]:.1f} km</b> '
                        f'at a detour cost of <b>{ea["detour_km"]:.0f} km</b>.</div>',unsafe_allow_html=True)
        elif elec_reroute and ea.get("has_alternative"):
            st.markdown(f'<div class="ok-box">✅ <b>Electrified-preference routing active.</b> '
                        f'Saved <b>{ea["elec_saving_km"]:.1f} km</b> of non-electrified track '
                        f'(detour: +{ea["detour_km"]:.0f} km). '
                        f'Remaining non-electrified: <b>{ea["penalised_ue_km"]:.1f} km</b> (unavoidable).</div>',unsafe_allow_html=True)
        elif ue>0:
            st.markdown(f'<div class="danger-box">🚫 <b>{ue:.1f} km non-electrified.</b> Electric vehicle cannot complete this route. '
                        f'No viable electrified detour exists within reasonable limits. '
                        f'Use a diesel vehicle or choose different stations.</div>',unsafe_allow_html=True)

    # KPI cards
    kc=st.columns(6)
    kc[0].markdown(kpi_card(f"{total_km:.1f} km","Route length"),unsafe_allow_html=True)
    kc[1].markdown(kpi_card(str(len(stops_x)),"Mandatory stops (X)"),unsafe_allow_html=True)
    kc[2].markdown(kpi_card(str(len(stops_r)),"Request stops (R)"),unsafe_allow_html=True)
    kc[3].markdown(kpi_card(f"{max_spd:.0f} km/h","Max speed"),unsafe_allow_html=True)
    kc[4].markdown(kpi_card(f"{max_grad:.0f} ‰","Max gradient"),unsafe_allow_html=True)
    kc[5].markdown(kpi_card(f"{elec_km:.0f} km",f"Electrified ({100*elec_km/total_km:.0f}%)",
                             delta=(f"⚠️ {ue_km:.0f} km non-electrified" if ue_km>0 else "✅ Fully electrified"),
                             color=(C["red"] if ue_km>0 else C["green"])),unsafe_allow_html=True)
    st.write("")

    # Main chart
    fig=make_profile_chart(df)
    st.plotly_chart(fig,use_container_width=True)

    # Electrification legend
    badges=""
    for sys in df["electrification"].unique():
        col=elec_color(sys); km_e=df[df["electrification"]==sys]["length_m"].sum()/1000
        icon="⚡" if sys!="NONE" else "🛢️"
        badges+=(f'<span style="background:{col};color:white;padding:3px 10px;'
                 f'border-radius:12px;font-size:.8rem;font-weight:600;margin:2px;display:inline-block">'
                 f'{icon} {sys} · {km_e:.1f} km</span> ')
    st.markdown(badges,unsafe_allow_html=True); st.write("")

    # Map
    map_df=df.dropna(subset=["lat","lon"])
    if not map_df.empty:
        with st.expander("🗺️ Route map",expanded=True):
            fig_m=make_route_map(df,st.session_state.via_ops or [],parser)
            st.plotly_chart(fig_m,use_container_width=True)

    # Gradient histogram + stats side by side
    with st.expander("📐 Gradient statistics"):
        sc1,sc2=st.columns([2,1])
        with sc1:
            g_vals=df["gradient_perm"]
            fig_gh=go.Figure()
            fig_gh.add_trace(go.Histogram(x=g_vals[g_vals>=0],name="Uphill",
                marker_color=C["red"],opacity=0.75,nbinsx=25))
            fig_gh.add_trace(go.Histogram(x=g_vals[g_vals<0],name="Downhill",
                marker_color=C["primary"],opacity=0.75,nbinsx=25))
            fig_gh.update_layout(barmode="overlay",height=280,
                xaxis_title="Gradient [‰]",yaxis_title="Segments",
                paper_bgcolor="white",plot_bgcolor="white",
                legend=dict(orientation="h"),margin=dict(t=10,b=40,l=50,r=10))
            st.plotly_chart(fig_gh,use_container_width=True)
        with sc2:
            st.metric("Max uphill",   f"{g_vals.max():.1f} ‰")
            st.metric("Max downhill", f"{g_vals.min():.1f} ‰")
            st.metric("RMS gradient", f"{(g_vals**2).mean()**0.5:.1f} ‰")
            st.metric("Avg gradient", f"{g_vals.mean():.1f} ‰")

    # Data table
    with st.expander("📋 Profile data table"):
        cols=["cum_km","station_name","stop_type","speed_kmh","gradient_perm",
              "electrification","recuperation","length_m","n_tracks","op_types"]
        cols=[c for c in cols if c in df.columns]
        ren=dict(cum_km="km",station_name="Waypoint",stop_type="Stop",
                 speed_kmh="Speed [km/h]",gradient_perm="Grad [‰]",
                 electrification="Electrification",recuperation="Recup.",
                 length_m="Length [m]",n_tracks="Tracks",op_types="Type")
        st.dataframe(df[cols].rename(columns=ren),use_container_width=True,
                     hide_index=True,height=350)
        st.download_button("⬇️ Download profile CSV",
                            df[cols].to_csv(index=False).encode(),
                            "track_profile.csv","text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – PROFILE EDITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.markdown("### ✏️ Interactive Profile Editor")
    df=st.session_state.profile_df

    # Via waypoints panel
    st.markdown('<div class="section-hdr">📍 Via Waypoints</div>',unsafe_allow_html=True)
    st.caption("Force the route through intermediate stations. Rebuild the profile after adding waypoints.")

    qv=st.text_input("Search waypoint to add",placeholder="e.g. Pardubice hl.n.",key="q_via")
    res_v=parser.search_stations(qv) if qv.strip() else []
    cva,cvb=st.columns([3,1])
    via_sel=None
    if res_v:
        via_sel=cva.selectbox("Select",options=[r[1] for r in res_v],
                               label_visibility="collapsed",key="via_opt")
    if cvb.button("➕ Add",use_container_width=True,disabled=not res_v):
        if via_sel:
            via_op=next((r[0] for r in res_v if r[1]==via_sel),None)
            if via_op and via_op not in st.session_state.via_ops:
                st.session_state.via_ops.append(via_op)
                st.rerun()

    if st.session_state.via_ops:
        st.markdown("**Via waypoints (in order):**")
        for i,vid in enumerate(st.session_state.via_ops):
            info=parser.op_info.get(vid,{})
            e1,e2,e3,e4=st.columns([0.3,0.3,3,0.8])
            e1.markdown(f"**{i+1}**")
            if e2.button("↑",key=f"up_{i}",disabled=i==0):
                st.session_state.via_ops[i],st.session_state.via_ops[i-1]=\
                st.session_state.via_ops[i-1],st.session_state.via_ops[i]; st.rerun()
            e3.markdown(f"📍 **{info.get('name',vid)}**  "
                        f"<span style='color:{C['grey']};font-size:.8rem'>{', '.join(info.get('types',[]))}</span>",
                        unsafe_allow_html=True)
            if e4.button("✕ Remove",key=f"rm_{i}",use_container_width=True):
                st.session_state.via_ops.pop(i); st.rerun()
        if st.button("🗑️ Clear all via waypoints"):
            st.session_state.via_ops=[]; st.rerun()
        st.markdown('<div class="info-box">ℹ️ Click <b>Build Track Profile</b> in the sidebar to apply waypoints.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No via-waypoints set. The router uses the direct shortest path.</div>',
                    unsafe_allow_html=True)

    # Map for visual waypoint context
    if df is not None:
        st.markdown('<div class="section-hdr">🗺️ Route Map</div>',unsafe_allow_html=True)
        st.caption("The map shows the current profile. Add via-waypoints above and rebuild to update.")
        fig_em=make_route_map(df,st.session_state.via_ops or [],parser,show_all_stops=True)
        st.plotly_chart(fig_em,use_container_width=True)

        # Override table
        st.markdown('<div class="section-hdr">📝 Speed & Stop Overrides</div>',unsafe_allow_html=True)
        st.caption("Edit speed limits or stop types directly. Changes apply only to the current simulation — "
                   "they do not change the underlying railML data.")

        edit_cols=["station_name","stop_type","speed_kmh","gradient_perm","electrification","length_m"]
        edit_cols=[c for c in edit_cols if c in df.columns]
        edited=st.data_editor(
            df[edit_cols].copy(),
            column_config={
                "station_name":   st.column_config.TextColumn("Waypoint",    disabled=True, width="large"),
                "stop_type":      st.column_config.SelectboxColumn("Stop",   options=["X","R",""],width="small"),
                "speed_kmh":      st.column_config.NumberColumn("Speed [km/h]",min_value=0,max_value=350,width="small"),
                "gradient_perm":  st.column_config.NumberColumn("Grad [‰]",  disabled=True,width="small"),
                "electrification":st.column_config.TextColumn("Electrif.",   disabled=True,width="medium"),
                "length_m":       st.column_config.NumberColumn("Length [m]", disabled=True,width="small"),
            },
            use_container_width=True,hide_index=True,num_rows="fixed",key="editor_table")

        if st.button("✅ Apply overrides",type="primary"):
            updated=df.copy()
            updated["speed_kmh"]=edited["speed_kmh"].values
            updated["stop_type"]=edited["stop_type"].values
            st.session_state.profile_df=updated
            st.session_state.rep_result=None
            st.success("✅ Profile updated. Run the simulation from the **▶️ Run** tab.")
    else:
        st.info("Build a track profile first (sidebar → **Build Track Profile**).")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – REPRESENTATIVE RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run_t:
    rep=st.session_state.rep_result
    if rep is None:
        st.info("👈 Build a profile, then click **▶️ Run** in the sidebar.")
        st.stop()

    hist=rep["hist"]; stats=rep["stats"]; snames=rep["stop_names"]
    _tr=rep["traction"]; _dd=rep["diesel_d"]; _eff=rep["efficiency"]
    consumed=to_unit(stats,_tr,_dd,_eff)
    df_p=st.session_state.profile_df
    tot_km=float(df_p["cum_km"].max()) if df_p is not None else 0
    avg_spd=tot_km/(stats["journey_time_s"]/3600) if stats["journey_time_s"]>0 else 0
    sn=df_p["station_name"].iloc[0] if df_p is not None else ""; en=df_p["station_name"].iloc[-1] if df_p is not None else ""

    st.markdown(f"### ▶️ {sn}  →  {en}   ·   {rep['vehicle_name']}")

    kc2=st.columns(6)
    kc2[0].markdown(kpi_card(fmt_dur(stats["journey_time_s"]),"Journey time"),unsafe_allow_html=True)
    kc2[1].markdown(kpi_card(f"{consumed:.1f}",f"Net consumed [{unit_lbl}]"),unsafe_allow_html=True)
    kc2[2].markdown(kpi_card(f"{stats['gross_kwh']:.1f}","Gross energy [kWh]"),unsafe_allow_html=True)
    kc2[3].markdown(kpi_card(f"{stats['regen_kwh']:.1f}","Recuperated [kWh]"),unsafe_allow_html=True)
    kc2[4].markdown(kpi_card(f"{avg_spd:.1f} km/h","Avg journey speed"),unsafe_allow_html=True)
    kc2[5].markdown(kpi_card(str(len(snames)),"Stops served"),unsafe_allow_html=True)
    st.write("")

    if snames:
        st.markdown("**Stops served:** " + "  →  ".join(f"*{s}*" for s in snames))
    else:
        st.caption("No intermediate stops served on this run.")

    if hist:
        fig3=make_subplots(rows=2,cols=1,shared_xaxes=True,
            subplot_titles=("Speed profile [km/h]","Cumulative energy [kWh]"),
            vertical_spacing=0.10,row_heights=[0.55,0.45])

        fig3.add_trace(go.Scatter(x=hist["km"],y=hist["v_limit_kmh"],name="Speed limit",
            line=dict(color=C["red"],dash="dot",width=1.8)),row=1,col=1)
        fig3.add_trace(go.Scatter(x=hist["km"],y=hist["v_kmh"],name="Actual speed",
            line=dict(color=C["primary"],width=2.5),fill="tozeroy",fillcolor=C["bg_blue"]),row=1,col=1)
        fig3.add_trace(go.Scatter(x=hist["km"],y=hist["gross_kwh"],name="Gross energy",
            line=dict(color=C["yellow"],width=2)),row=2,col=1)
        fig3.add_trace(go.Scatter(x=hist["km"],y=hist["regen_kwh"],name="Recuperated",
            line=dict(color=C["green"],width=2,dash="dash")),row=2,col=1)
        fig3.add_trace(go.Scatter(x=hist["km"],y=hist["net_kwh"],name="Net energy",
            line=dict(color=C["secondary"],width=2.5)),row=2,col=1)

        if df_p is not None:
            for _,sr in df_p[df_p["stop_type"]=="X"].iterrows():
                fig3.add_vline(x=sr["cum_km"],line_width=0.7,line_dash="dot",
                               line_color="#CBD5E1",row=1,col=1)

        fig3.update_xaxes(title_text="Distance from departure [km]",row=2,col=1,gridcolor=C["light"])
        fig3.update_yaxes(title_text="Speed [km/h]",row=1,col=1,gridcolor=C["light"])
        fig3.update_yaxes(title_text="Energy [kWh]",row=2,col=1,gridcolor=C["light"])
        fig3.update_layout(height=580,paper_bgcolor="white",plot_bgcolor="white",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,
                        bgcolor="rgba(255,255,255,0.9)",bordercolor="#E2E8F0",borderwidth=1),
            margin=dict(t=70,b=10,l=60,r=20),font=dict(family="Inter,sans-serif",size=12))
        st.plotly_chart(fig3,use_container_width=True)

        if stats["gross_kwh"]>0 and tot_km>0:
            rp=stats["regen_kwh"]/stats["gross_kwh"]*100; spec=stats["net_kwh"]/tot_km
            st.markdown("---")
            ec1,ec2,ec3,ec4=st.columns(4)
            ec1.metric("Recuperation share",f"{rp:.1f}%")
            ec2.metric("Specific net energy",f"{spec:.3f} kWh/km")
            if _tr=="DIESEL":
                ec3.metric("Specific fuel",f"{consumed/tot_km*1000:.1f} mL/km")
                ec4.metric("CO₂ (est.)",f"{consumed*2.65:.0f} kg",
                            help="Diesel combustion: ~2.65 kg CO₂/L")
            else:
                ec3.metric("Net kWh/km",f"{spec:.3f}")
                ec4.metric("Regen saved",f"{stats['regen_kwh']:.1f} kWh")

        with st.expander("📊 Detailed distributions"):
            dd1,dd2=st.columns(2)
            with dd1:
                fig_sv=go.Figure(go.Histogram(x=hist["v_kmh"],nbinsx=40,
                    marker_color=C["primary"],opacity=0.8))
                fig_sv.update_layout(title="Time distribution at speed",
                    xaxis_title="Speed [km/h]",yaxis_title="Seconds",height=300,
                    paper_bgcolor="white",plot_bgcolor="white",
                    margin=dict(t=40,b=40,l=50,r=10))
                st.plotly_chart(fig_sv,use_container_width=True)
            with dd2:
                wf_vals=[stats["gross_kwh"],-stats["regen_kwh"],stats["net_kwh"]]
                fig_wf=go.Figure(go.Waterfall(
                    orientation="v",measure=["absolute","relative","total"],
                    x=["Gross consumed","Recuperated","Net consumed"],y=wf_vals,
                    connector=dict(line=dict(color="#CBD5E1")),
                    decreasing=dict(marker=dict(color=C["green"])),
                    increasing=dict(marker=dict(color=C["red"])),
                    totals=dict(marker=dict(color=C["primary"])),
                    text=[f"{v:.1f} kWh" for v in wf_vals],textposition="outside"))
                fig_wf.update_layout(title="Energy waterfall [kWh]",height=300,
                    paper_bgcolor="white",plot_bgcolor="white",
                    margin=dict(t=40,b=40,l=50,r=10))
                st.plotly_chart(fig_wf,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mc_t:
    mc_df=st.session_state.mc_result
    if mc_df is None:
        st.info("👈 Build a profile, then click **🎲 MC** in the sidebar.")
        # Show stop type breakdown as hint
        df_p=st.session_state.profile_df
        if df_p is not None:
            track=TrackProfile(df_p)
            if track.n_request==0:
                st.markdown('<div class="warn-box">⚠️ <b>No request stops (R) on this route.</b> '
                            'All stops are mandatory (X), so Monte Carlo variance will be zero. '
                            'The current route has only <b>station</b>-type OPs (mandatory X). '
                            'Try a route with smaller stopping points (type <i>stoppingPoint</i> = R) '
                            'for meaningful stochastic results.</div>',unsafe_allow_html=True)
        st.stop()

    ul=mc_df["unit"].iloc[0]
    df_p=st.session_state.profile_df
    route_n=(f"{df_p['station_name'].iloc[0]} → {df_p['station_name'].iloc[-1]}" if df_p is not None else "")

    st.markdown(f"### 🎲 Monte Carlo — {route_n}")

    # Stop type summary
    if df_p is not None:
        track=TrackProfile(df_p)
        nm=track.n_mandatory; nr=track.n_request
        if nr==0:
            st.markdown('<div class="warn-box">⚠️ All stops are mandatory (X) — Monte Carlo std will be zero. '
                        'Try a route with stoppingPoint-type halts for stochastic variation.</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">ℹ️ Route has <b>{nm} mandatory (X)</b> and '
                        f'<b>{nr} request (R)</b> stops. MC sweeps the probability of serving request stops.</div>',
                        unsafe_allow_html=True)
    st.caption(f"N = {int(mc_n)} simulation runs per probability level")

    # Summary table
    disp=mc_df.copy()
    disp["Mean time"]=disp["mean_t"].apply(fmt_dur)
    disp["Fastest"]=disp["min_t"].apply(fmt_dur)
    disp["Slowest"]=disp["max_t"].apply(fmt_dur)
    show_cols={"prob":f"Probability","mean_e":f"Mean [{ul}]","std_e":f"Std [{ul}]",
               "min_e":f"Min [{ul}]","max_e":f"Max [{ul}]",
               "savings":f"Savings [{ul}]","Mean time":"Mean time",
               "Fastest":"Fastest","Slowest":"Slowest"}
    st.dataframe(disp.rename(columns=show_cols)[[v for v in show_cols.values() if v in disp.rename(columns=show_cols).columns]],
                 use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download CSV",mc_df.to_csv(index=False).encode(),"mc_results.csv","text/csv")
    st.markdown("---")

    # Charts
    mc1,mc2=st.columns(2)
    with mc1:
        fig_bar=go.Figure(go.Bar(x=mc_df["prob"],y=mc_df["savings"],
            marker=dict(color=mc_df["savings"],colorscale="Blues",showscale=True,
                        colorbar=dict(title=ul,len=0.8)),
            text=mc_df["savings"].round(2),texttemplate="%{text}",textposition="outside",
            hovertemplate="<b>%{x}</b><br>Savings: %{y:.2f} "+ul+"<extra></extra>"))
        fig_bar.update_layout(title=f"<b>Energy savings vs. all-stops</b> [{ul}]",
            xaxis_title="Request-stop probability",yaxis_title=f"Savings [{ul}]",
            height=400,paper_bgcolor="white",plot_bgcolor="white",
            margin=dict(t=60,b=50,l=60,r=20),font=dict(family="Inter,sans-serif",size=12),
            yaxis=dict(gridcolor=C["light"]))
        st.plotly_chart(fig_bar,use_container_width=True)

    with mc2:
        fig_err=go.Figure()
        fig_err.add_trace(go.Scatter(
            x=list(mc_df["prob"])+list(mc_df["prob"].iloc[::-1]),
            y=list(mc_df["max_e"])+list(mc_df["min_e"].iloc[::-1]),
            fill="toself",fillcolor=C["bg_blue"],
            line=dict(color="rgba(255,255,255,0)"),name="Min–Max range"))
        fig_err.add_trace(go.Scatter(
            x=list(mc_df["prob"])+list(mc_df["prob"].iloc[::-1]),
            y=list(mc_df["mean_e"]+mc_df["std_e"])+list((mc_df["mean_e"]-mc_df["std_e"]).iloc[::-1]),
            fill="toself",fillcolor="rgba(37,99,235,0.18)",
            line=dict(color="rgba(255,255,255,0)"),name="Mean ± 1σ"))
        fig_err.add_trace(go.Scatter(x=mc_df["prob"],y=mc_df["mean_e"],
            mode="markers+lines",
            marker=dict(size=11,color=C["primary"],line=dict(color="white",width=2)),
            line=dict(color=C["primary"],width=2.5),name="Mean"))
        fig_err.update_layout(title=f"<b>Energy distribution</b> [{ul}]",
            xaxis_title="Request-stop probability",yaxis_title=f"Energy [{ul}]",
            height=400,paper_bgcolor="white",plot_bgcolor="white",
            margin=dict(t=60,b=50,l=60,r=20),font=dict(family="Inter,sans-serif",size=12),
            yaxis=dict(gridcolor=C["light"]),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_err,use_container_width=True)

    # Journey time
    fig_t=go.Figure()
    fig_t.add_trace(go.Scatter(
        x=list(mc_df["prob"])+list(mc_df["prob"].iloc[::-1]),
        y=list(mc_df["max_t"]/60)+list(mc_df["min_t"].iloc[::-1]/60),
        fill="toself",fillcolor=C["bg_orange"],line=dict(color="rgba(255,255,255,0)"),name="Min–Max"))
    fig_t.add_trace(go.Scatter(x=mc_df["prob"],y=mc_df["mean_t"]/60,
        mode="markers+lines",
        marker=dict(size=11,color=C["accent"],line=dict(color="white",width=2)),
        line=dict(color=C["accent"],width=2.5),name="Mean time"))
    fig_t.update_layout(title="<b>Journey time vs. stopping policy</b>",
        xaxis_title="Request-stop probability",yaxis_title="Journey time [min]",
        height=360,paper_bgcolor="white",plot_bgcolor="white",
        margin=dict(t=60,b=50,l=60,r=20),font=dict(family="Inter,sans-serif",size=12),
        yaxis=dict(gridcolor=C["light"]),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig_t,use_container_width=True)

    with st.expander("📊 Energy–time trade-off"):
        fig_et=go.Figure(go.Scatter(
            x=mc_df["mean_t"]/60,y=mc_df["mean_e"],mode="markers+text",
            marker=dict(size=mc_df["savings"].clip(lower=0)*3+14,
                        color=mc_df["savings"],colorscale="RdYlGn",showscale=True,
                        colorbar=dict(title=f"Savings [{ul}]"),
                        line=dict(color="white",width=1.5)),
            text=mc_df["prob"],textposition="top center",
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f} min<br>Energy: %{y:.2f} "+ul+"<extra></extra>"))
        fig_et.update_layout(title="<b>Energy–time trade-off</b> by stopping policy",
            xaxis_title="Mean journey time [min]",yaxis_title=f"Mean energy [{ul}]",
            height=420,paper_bgcolor="white",plot_bgcolor="white",
            margin=dict(t=60,b=50,l=60,r=20),font=dict(family="Inter,sans-serif",size=12))
        st.plotly_chart(fig_et,use_container_width=True)