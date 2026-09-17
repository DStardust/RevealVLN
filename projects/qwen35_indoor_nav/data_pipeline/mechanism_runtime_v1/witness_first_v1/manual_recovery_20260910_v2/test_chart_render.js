'use strict';
// Main-agent JavaScript/DOM smoke test, not a browser screenshot/layout test.
const fs = require('fs'), vm = require('vm'), cp = require('child_process'), assert = require('assert');
const root = '/mnt/data_nas/deeprobotics/daiyang/vla';
const chart = root + '/projects/qwen35_indoor_nav/sft_acceptance/monitor_charts_v1';
const proc = cp.spawnSync(root+'/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3', ['-I','-S','-B',chart+'/server.py','--check'], {encoding:'utf8',timeout:15000,maxBuffer:4*1024*1024});
assert.strictEqual(proc.status,0,proc.stderr);
const data = JSON.parse(proc.stdout);
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.style = {}; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(k,v) { this.attributes[k] = String(v); }
  querySelectorAll(tag) { return this.children.flatMap(c => c instanceof Element ? [...(c.tag === tag ? [c] : []), ...c.querySelectorAll(tag)] : []); }
}
const nodes = new Map();
const document = {getElementById(id) { if (!nodes.has(id)) nodes.set(id,new Element('div')); return nodes.get(id); }, createElement: tag => new Element(tag), createElementNS: (_,tag) => new Element(tag)};
let script = fs.readFileSync(chart+'/index.html','utf8').match(/<script>([\s\S]*)<\/script>/)[1];
assert(script.includes('refresh();setInterval(refresh,10000);'));
script = script.replace('refresh();setInterval(refresh,10000);','');
vm.runInNewContext(script+'\nrender(DATA);', {document, DATA:data, Date, Number, Object, Math, JSON, Error}, {timeout:5000});
assert.strictEqual(nodes.get('cards').children.length,6);
assert(nodes.get('loss').querySelectorAll('path').length >= 2);
assert(nodes.get('throughput').querySelectorAll('path').length >= 1);
assert.strictEqual(nodes.get('matrix').querySelectorAll('tr').length,5);
assert(nodes.get('queues').querySelectorAll('tr').length > 5);
for (const node of nodes.values()) for (const tag of ['path','rect','circle']) for (const item of node.querySelectorAll(tag)) {
  assert(!/NaN|Infinity|undefined/.test(JSON.stringify(item.attributes)), 'nonfinite chart geometry');
}
console.log(JSON.stringify({status:'LIVE_DATA_JS_DOM_RENDER_PASS',browser_layout_tested:false,points:data.points.length,live_production_queues:data.queues.filter(q=>q.alive).length,decisions:data.decisions,updates:data.progress.data.cursor.updates}));
