let refreshBusy=false;
async function refresh(){
 if(refreshBusy)return;
 refreshBusy=true;
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
 try{
  const response=await fetch(new URL('api/status',document.baseURI),{cache:'no-store',signal:controller.signal});
  if(!response.ok)throw Error('HTTP '+response.status);
  const d=await response.json();render(d);renderTargeted(d);
  $('connection').textContent='已连接 · '+new Date(d.server_unix*1000).toLocaleTimeString();
  $('connection').className='pill good';
 }catch(e){
  $('connection').textContent='连接失败 · 保留上次视图，自动重试';
  $('connection').className='pill bad';
  $('errors').textContent='监控读取失败：'+e.message+'。请检查 SSH 转发；不代表训练或评测停止。';
 }finally{clearTimeout(timer);refreshBusy=false;setTimeout(refresh,10000)}
}
refresh();
