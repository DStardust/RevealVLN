// Browserless DOM contract test. This does not claim a browser visual inspection.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path'),root=__dirname;
const source=fs.readFileSync(path.join(root,'refresh.js'),'utf8');
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
class Node {
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.attrs={};}
 replaceChildren(...v){this.children=v;}
 appendChild(v){this.children.push(v);return v;}
 setAttribute(k,v){this.attrs[k]=v;}
}
function fixture(){
 const arm=()=>({progress:null,result:null,acceptance:null,launch:null,resource:null,live_rank_pids:[],history:[]});
 const evaluation=()=>({progress:null,result:null,launch:null,worker_live:false,worker_pid:null,checkpoint_bound:false,review:null,failure:null});
 return {monitor_version:'ordinary_history8_pair_r2',paired_history:{
  arms:{control_recent2:arm(),treatment_prefix8:arm()},
  first200:{control_recent2:null,treatment_prefix8:null},
  evaluation:{control_recent2:evaluation(),treatment_prefix8:evaluation()},
  workflow:{live:true,pid:123,status:{status:'WAIT_FIXED4000_TRAINING_ACCEPTANCE',arm:'control_recent2',unix:1}},
  prior_memory_training:{updates:4},collected_unix:1000,comparison:null
 },sr40:{historical_full:{sr:.22,spl:.18},best_internal:{sr:.21,spl:.18}},gpu:{devices:[]}};
}
async function run(d){
 const nodes=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Node('div')]));
 const doc={getElementById:id=>{assert(nodes.has(id),'missing DOM id '+id);return nodes.get(id);},createElement:tag=>new Node(tag),createElementNS:(_ns,tag)=>new Node(tag)};
 vm.runInNewContext(source,{document:doc,fetch:async()=>({ok:true,json:async()=>d}),setInterval:()=>{},Date,Math,Number,Object,String,Error});
 await new Promise(resolve=>setImmediate(resolve));
 assert.strictEqual(nodes.get('error').textContent,'');
 return nodes;
}
function cells(nodes,id,row){return nodes.get(id).children[row].children.map(x=>x.textContent);}
(async()=>{
 let d=fixture(),nodes=await run(d);
 assert.deepStrictEqual(cells(nodes,'navigation',2),['内部100条','2帧对照 R1','未测','未测']);
 assert(nodes.get('eval-live').textContent.includes('进程存活 PID 123'));
 assert(!nodes.get('eval-live').textContent.includes('心跳超过'));
 d=fixture();
 d.paired_history.evaluation.control_recent2.progress={completed:99,total:100};
 d.paired_history.evaluation.control_recent2.result={status:'COMPLETE',completed:99,sr:.9,spl:.8};
 d.paired_history.evaluation.control_recent2.worker_live=true;
 d.paired_history.evaluation.control_recent2.worker_pid=456;
 nodes=await run(d);
 assert.strictEqual(cells(nodes,'navigation',2)[2],'未测');
 assert.strictEqual(cells(nodes,'evaluation',0)[2],'99 / 100');
 assert.notStrictEqual(cells(nodes,'evaluation',0)[4],'完整指标已审核');
 d.paired_history.evaluation.control_recent2.result={status:'COMPLETE',completed:100,sr:.24,spl:.20};
 d.paired_history.comparison={selected_candidate_for_separate_full_protocol:'control_recent2'};
 nodes=await run(d);
 assert.strictEqual(cells(nodes,'navigation',2)[2],'24.00%');
 assert(nodes.get('phase').textContent.includes('待完整基准'));
 assert(nodes.get('comparison').textContent.includes('SR40仍未验收'));
 console.log('PASS_DOM_CONTRACT_PENDING_PARTIAL_COMPLETE; NO_BROWSER_VISUAL_TEST');
})().catch(e=>{process.stderr.write(String(e.stack)+'\n');process.exitCode=1;});
