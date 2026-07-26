/* Figures for esmfold2.html: the diagrams of the ESMFold2 walkthrough, built
   into the empty containers the figure partials leave for them.

   Ported from the standalone illustrated page. Its own page chrome is gone:
   the scroll-progress bar, the floating section nav and the theme toggle all
   belong to templates/post.html here, so what remains is the figure builders,
   the scroll-reveal observer and the glossary tooltip. No dependencies. */
(function(){

    var io=new IntersectionObserver(function(es){
      es.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } });
    },{rootMargin:'0px 0px -8% 0px',threshold:.08});
    document.querySelectorAll('.reveal').forEach(function(el){io.observe(el);});

    /* ---- tensor glyphs: cubes (3D), matrices (2D), vectors (1D) ---- */
    function wrap(vw,vh,g){return '<svg viewBox="0 0 '+vw+' '+vh+'" role="img" aria-hidden="true">'+g+'</svg>';}
    function ln(x1,y1,x2,y2,col,op){return '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+col+'" stroke-opacity="'+op+'" stroke-width="1"/>';}
    function poly(pts,fill,stroke){return '<polygon points="'+pts+'" fill="'+fill+'" stroke="'+stroke+'" stroke-width="1.4" stroke-linejoin="round"/>';}
    function txt(x,y,s,o){o=o||{};var a=o.anchor||'middle',sz=o.size||9.5,fl=o.fill||'var(--ink-soft)';
      var w=o.weight?' font-weight="'+o.weight+'"':'';var tr=o.rotate?' transform="rotate('+o.rotate+' '+x+' '+y+')"':'';
      return '<text x="'+x+'" y="'+y+'" font-size="'+sz+'" fill="'+fl+'" text-anchor="'+a+'"'+w+tr+'>'+s+'</text>';}
    function mix(col,p){return "color-mix(in srgb, "+col+" "+p+"%, var(--surface))";}

    /* gentle log-compressed axis sizing: big dims read larger without exploding */
    function refVal(s){ if(/^\d+$/.test(s)) return parseInt(s,10); var m={L:320,N_atom:2400}; return m[s]!=null?m[s]:300; }
    function scale(v,lo,hi){ var a=Math.log10(3),b=Math.log10(2600);
      v=Math.max(3,Math.min(2600,v)); var t=(Math.log10(v)-a)/(b-a); return Math.round(lo+t*(hi-lo)); }
    var VH=132; /* shared viewBox height → all glyphs render at one pixel height */

    function cubeSVG(name,d,c){
      var col='var(--'+c+')';
      var W=scale(refVal(d[0]),34,60), H=scale(refVal(d[1]),34,60), dep=scale(refVal(d[2]),12,26);
      var dx=dep, dy=-Math.round(dep*0.8), px=24, py=28-dy, g='';   /* top-back edge fixed at y=28 */
      function P(a){return a.map(function(p){return p[0]+','+p[1];}).join(' ');}
      g+=poly(P([[px,py],[px+W,py],[px+W+dx,py+dy],[px+dx,py+dy]]),mix(col,22),col);       /* top  */
      g+=poly(P([[px+W,py],[px+W+dx,py+dy],[px+W+dx,py+H+dy],[px+W,py+H]]),mix(col,55),col);/* side */
      g+=poly(P([[px,py],[px+W,py],[px+W,py+H],[px,py+H]]),mix(col,38),col);                /* face */
      g+=ln(px+W/3,py,px+W/3,py+H,col,.32)+ln(px+2*W/3,py,px+2*W/3,py+H,col,.32);
      g+=ln(px,py+H/3,px+W,py+H/3,col,.32)+ln(px,py+2*H/3,px+W,py+2*H/3,col,.32);
      g+=txt(px+(W+dx)/2,12,name,{anchor:'middle',size:10.5,weight:700,fill:col});          /* name band */
      g+=txt(px+W/2,py+H+15,d[0]||'',{});                                                   /* width  */
      g+=txt(px-11,py+H/2,d[1]||'',{rotate:-90});                                           /* height */
      g+=txt(px+W+dx+11,py+dy+H/2,d[2]||'',{rotate:-90,size:9});                            /* depth  */
      return wrap(px+W+dx+22,VH,g);
    }
    function matSVG(name,d,c){
      var col='var(--'+c+')';
      var W=scale(refVal(d[1]),30,58), H=scale(refVal(d[0]),34,62), px=24, py=30, g='',i;
      g+='<rect x="'+px+'" y="'+py+'" width="'+W+'" height="'+H+'" rx="1.5" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.4"/>';
      g+=ln(px+W/2,py,px+W/2,py+H,col,.4);
      for(i=1;i<4;i++){g+=ln(px,py+H*i/4,px+W,py+H*i/4,col,.4);}
      g+=txt(px+W/2,12,name,{anchor:'middle',size:10.5,weight:700,fill:col});
      g+=txt(px+W/2,py+H+15,d[1]||'',{});
      g+=txt(px-11,py+H/2,d[0]||'',{rotate:-90});
      return wrap(px+W+22,VH,g);
    }
    function vecSVG(name,d,c){
      var col='var(--'+c+')';
      var W=15, H=scale(refVal(d[0]),40,64), px=26, py=30, g='',i;
      g+='<rect x="'+px+'" y="'+py+'" width="'+W+'" height="'+H+'" rx="1.5" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.4"/>';
      for(i=1;i<5;i++){g+=ln(px,py+H*i/5,px+W,py+H*i/5,col,.4);}
      g+=txt(px+W/2,12,name,{anchor:'middle',size:10.5,weight:700,fill:col});
      g+=txt(px-11,py+H/2,d[0]||'',{rotate:-90});
      return wrap(px+W+22,VH,g);
    }
    /* operation pictograms: depict what each op does */
    function ic(w,g){return '<svg viewBox="0 0 '+w+' 40" role="img" aria-hidden="true">'+g+'</svg>';}
    function cir(x,y,r,f,s){return '<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+(f||'none')+'"'+(s?' stroke="'+s+'" stroke-width="1.3"':'')+'/>';}
    function pth(d,s,w){return '<path d="'+d+'" fill="none" stroke="'+s+'" stroke-width="'+(w||1.6)+'" stroke-linecap="round" stroke-linejoin="round"/>';}
    function rct(x,y,w,h,f,s){return '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="1.4" fill="'+(f||'none')+'"'+(s?' stroke="'+s+'" stroke-width="1.3"':'')+'/>';}
    function ah(x,y){return '<path d="M'+(x-4)+' '+(y-3)+' L'+x+' '+y+' L'+(x-4)+' '+(y+3)+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>';}
    function opIcon(op,c){
      var A='var(--'+c+')',K='var(--ink-soft)',g='',i;
      if(op==='frozen-stack'){
        for(i=0;i<4;i++) g+=rct(8,9+i*7,22,4,mix(A,28),A);
        g+='<g stroke="'+A+'" stroke-width="1.2" stroke-linecap="round"><line x1="37" y1="4" x2="37" y2="14"/><line x1="32" y1="9" x2="42" y2="9"/><line x1="33.5" y1="5.5" x2="40.5" y2="12.5"/><line x1="40.5" y1="5.5" x2="33.5" y2="12.5"/></g>';
        return ic(46,g);
      }
      if(op==='combine'){
        var hs=[9,17,13,21];
        for(i=0;i<4;i++) g+=rct(6+i*5,34-hs[i],3.4,hs[i],mix(A,35),A);
        return ic(44,g+ah(30,20)+rct(34,11,5,23,mix(A,48),A));
      }
      if(op==='loop'){
        g+=pth('M23 8 A12 12 0 1 1 12 25',A,1.9);
        return ic(44,g+'<path d="M8 21 L12 26 L17 23" fill="none" stroke="'+A+'" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>');
      }
      if(op==='denoise'){
        var dp=[[8,12],[6,22],[11,27],[13,15],[9,33]];
        for(i=0;i<dp.length;i++) g+=cir(dp[i][0],dp[i][1],1.7,K);
        g+='<line x1="18" y1="20" x2="25" y2="20" stroke="'+K+'" stroke-width="1.3"/>'+ah(26,20);
        return ic(44,g+cir(35,12,1.9,A)+cir(35,20,1.9,A)+cir(35,28,1.9,A));
      }
      if(op==='gauge'){
        g+=pth('M7 31 A15 15 0 0 1 37 31',K,1.6);
        return ic(44,g+'<line x1="22" y1="31" x2="31" y2="16" stroke="'+A+'" stroke-width="1.9" stroke-linecap="round"/>'+cir(22,31,2,A));
      }
      if(op==='window'){
        for(i=0;i<7;i++){var wx=5+i*5,inw=(i>=2&&i<=4);g+=rct(wx,17,4,9,inw?mix(A,38):'var(--surface)',inw?A:K);}
        return ic(44,g+'<path d="M14 13 L14 11 L30 11 L30 13" fill="none" stroke="'+A+'" stroke-width="1.4"/>');
      }
      if(op==='concat'){
        return ic(44,rct(5,14,6,12,mix(A,30),A)+rct(12,14,6,12,mix(A,15),A)+ah(28,20)+rct(32,14,8,12,mix(A,30),A));
      }
      if(op==='outer'){
        g+=rct(13,12,22,20,'none',mix(A,45));
        g+='<line x1="20" y1="12" x2="20" y2="32" stroke="'+mix(A,45)+'" stroke-width="1"/><line x1="27" y1="12" x2="27" y2="32" stroke="'+mix(A,45)+'" stroke-width="1"/><line x1="13" y1="19" x2="35" y2="19" stroke="'+mix(A,45)+'" stroke-width="1"/><line x1="13" y1="26" x2="35" y2="26" stroke="'+mix(A,45)+'" stroke-width="1"/>';
        return ic(44,g+rct(7,12,4,20,mix(A,40),A)+rct(13,6,22,4,mix(A,40),A));
      }
      if(op==='project'){
        return ic(46,poly('8,9 34,17 34,23 8,31',mix(A,25),A)+ah(41,20));
      }
      if(op==='tri-out'||op==='tri-in'){
        g+=pth('M22 9 L11 31 L33 31 Z',A,1.6)+cir(22,9,2.2,A)+cir(11,31,2.2,A)+cir(33,31,2.2,A);
        if(op==='tri-out') g+='<path d="M27 17 A9 9 0 0 1 27 25" fill="none" stroke="'+A+'" stroke-width="1.5"/><path d="M25.5 23.5 L27.5 25.5 L29 22.5" fill="none" stroke="'+A+'" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
        else g+='<path d="M17 25 A9 9 0 0 1 17 17" fill="none" stroke="'+A+'" stroke-width="1.5"/><path d="M18.5 18.5 L16.5 16.5 L15 19.5" fill="none" stroke="'+A+'" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
        return ic(44,g);
      }
      if(op==='bottleneck'){
        g+=rct(8,14,5,12,mix(A,28),A)+rct(18,7,7,26,mix(A,42),A)+rct(30,14,5,12,mix(A,28),A);
        return ic(44,g+'<line x1="13" y1="20" x2="18" y2="20" stroke="'+K+'" stroke-width="1.2"/><line x1="25" y1="20" x2="30" y2="20" stroke="'+K+'" stroke-width="1.2"/>');
      }
      if(op==='attn'){
        var ky=[9,17,25,33];
        for(i=0;i<4;i++) g+='<line x1="34" y1="21" x2="9" y2="'+ky[i]+'" stroke="'+K+'" stroke-width="1" stroke-opacity=".7"/>';
        for(i=0;i<4;i++) g+=cir(8,ky[i],1.8,mix(A,50),A);
        return ic(44,g+cir(35,21,2.6,A));
      }
      if(op==='scatter'){
        var ty=[8,16,24,32];
        for(i=0;i<4;i++) g+='<line x1="10" y1="20" x2="34" y2="'+ty[i]+'" stroke="'+K+'" stroke-width="1" stroke-opacity=".7"/>';
        g+=cir(9,20,2.6,A);
        for(i=0;i<4;i++) g+=cir(35,ty[i],1.8,K);
        return ic(44,g);
      }
      if(op==='sum'){
        g+='<circle cx="22" cy="20" r="12" fill="none" stroke="'+A+'" stroke-width="1.6"/>';
        g+='<line x1="22" y1="13" x2="22" y2="27" stroke="'+A+'" stroke-width="1.8" stroke-linecap="round"/>';
        g+='<line x1="15" y1="20" x2="29" y2="20" stroke="'+A+'" stroke-width="1.8" stroke-linecap="round"/>';
        return ic(44,g);
      }
      return miniSVG('cube',c);
    }
    function miniSVG(kind,c){        /* shape-only icon of a tensor, for use inside op blocks */
      var col='var(--'+c+')',g='';
      function P(a){return a.map(function(p){return p[0]+','+p[1];}).join(' ');}
      if(kind==='cube'){
        var W=26,H=26,dx=10,dy=-8,px=3,py=13;
        g+=poly(P([[px,py],[px+W,py],[px+W+dx,py+dy],[px+dx,py+dy]]),mix(col,22),col);
        g+=poly(P([[px+W,py],[px+W+dx,py+dy],[px+W+dx,py+H+dy],[px+W,py+H]]),mix(col,55),col);
        g+=poly(P([[px,py],[px+W,py],[px+W,py+H],[px,py+H]]),mix(col,38),col);
        return wrap(px+W+dx+3,46,g);
      }
      if(kind==='matrix'){
        var mW=20,mH=30,mx=3,my=8;
        g+='<rect x="'+mx+'" y="'+my+'" width="'+mW+'" height="'+mH+'" rx="1.5" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.3"/>';
        g+=ln(mx+mW/2,my,mx+mW/2,my+mH,col,.4)+ln(mx,my+mH/2,mx+mW,my+mH/2,col,.4);
        return wrap(mx+mW+3,46,g);
      }
      var vW=10,vH=30,vx=4,vy=8;
      g+='<rect x="'+vx+'" y="'+vy+'" width="'+vW+'" height="'+vH+'" rx="1.5" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.3"/>';
      return wrap(vx+vW+3,46,g);
    }
    document.querySelectorAll('.tviz').forEach(function(n){
      var k=n.getAttribute('data-kind'),nm=n.getAttribute('data-name')||'',
          d=(n.getAttribute('data-dims')||'').split('|'),c=n.getAttribute('data-c')||'pair';
      if(n.classList.contains('mini')){ var op=n.getAttribute('data-op'); n.innerHTML=op?opIcon(op,c):miniSVG(k,c); return; }
      n.innerHTML = k==='cube'?cubeSVG(nm,d,c) : k==='matrix'?matSVG(nm,d,c) : vecSVG(nm,d,c);
    });

    // Reserve exactly each side branch's height (they float; add nothing to the pipeline row).
    function branchMax(sel){var h=0;document.querySelectorAll(sel).forEach(function(b){h=Math.max(h,b.getBoundingClientRect().height);});return h;}
    function fitBranches(){
      var top=document.querySelector('.branchspace.top'), bot=document.querySelector('.branchspace.bot');
      if(top){var u=branchMax('.branch.up');top.style.height=u?Math.ceil(u+18)+'px':'0';}
      if(bot){var d=branchMax('.branch.down');bot.style.height=d?Math.ceil(d+18)+'px':'0';}
    }
    fitBranches();
    window.addEventListener('resize',fitBranches,{passive:true});

    // Figure 2: hand-laid overview built around the persistent pair state z.
    // Three input→encoder chains WRITE INTO a central z hub (which the Parcae
    // recurrence refines ×T); diffusion READS OUT of z. Glyphs colored by
    // stream (matching the legend); section signposts replace per-node badges.
    function buildFig0(){
      var INK='var(--ink-faint)';
      var C={seq:'var(--seq)',esmc:'var(--esmc)',pair:'var(--pair)',atom:'var(--atom)',conf:'var(--conf)'};
      var SID={inputs:'inputs',esmc:'esmc',parcae:'parcae',diffusion:'diffusion',confidence:'confidence'};
      function T(x,y,t,cv,sz,w,fam,anchor){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||10)+'" fill="'+cv+'" text-anchor="'+(anchor||'middle')+'"'+(w?' font-weight="'+w+'"':'')+' font-family="'+(fam||'var(--mono)')+'">'+t+'</text>';}
      function cube(x,y,w,h,dep,name,dims,col,dash){
        var dx=dep,dy=-Math.round(dep*0.8),d=dash?' stroke-dasharray="4 3"':'',s='';
        s+='<path d="M'+x+' '+y+' h'+w+' l'+dx+' '+dy+' h'+(-w)+' Z" fill="'+mix(col,20)+'" stroke="'+col+'" stroke-width="1.4"'+d+'/>';
        s+='<path d="M'+(x+w)+' '+y+' l'+dx+' '+dy+' v'+h+' l'+(-dx)+' '+(-dy)+' Z" fill="'+mix(col,50)+'" stroke="'+col+'" stroke-width="1.4"'+d+'/>';
        s+='<path d="M'+x+' '+y+' h'+w+' v'+h+' h'+(-w)+' Z" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.4"'+d+'/>';
        s+=T(x+w/2+dx/2,y+dy-7,name,col,10.5,700,'var(--sans)');           /* name clears the projected top */
        if(dims) s+=T(x+w/2+dx/2,y+h+13,dims,INK,8);
        return s;
      }
      function mat(x,y,w,h,name,dims,col){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2" fill="'+mix(col,30)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<line x1="'+(x+w/2)+'" y1="'+y+'" x2="'+(x+w/2)+'" y2="'+(y+h)+'" stroke="'+col+'" stroke-opacity=".45"/>';
        s+='<line x1="'+x+'" y1="'+(y+h/2)+'" x2="'+(x+w)+'" y2="'+(y+h/2)+'" stroke="'+col+'" stroke-opacity=".45"/>';
        s+=T(x+w/2,y-7,name,col,10.5,700,'var(--sans)'); if(dims) s+=T(x+w/2,y+h+13,dims,INK,8);
        return s;
      }
      function vec(x,y,w,h,name,dims,col,cells,row){               /* 1-D vector: a length-L strip of cells (row or column) */
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2" fill="'+mix(col,30)+'" stroke="'+col+'" stroke-width="1.4"/>';
        var n=cells||5,i;
        for(i=1;i<n;i++){
          if(row){var xx=x+w*i/n;s+='<line x1="'+xx+'" y1="'+y+'" x2="'+xx+'" y2="'+(y+h)+'" stroke="'+col+'" stroke-opacity=".45"/>';}
          else   {var yy=y+h*i/n;s+='<line x1="'+x+'" y1="'+yy+'" x2="'+(x+w)+'" y2="'+yy+'" stroke="'+col+'" stroke-opacity=".45"/>';}
        }
        s+=T(x+w/2,y-7,name,col,10.5,700,'var(--sans)');
        if(dims){ if(row) s+=T(x+w/2,y+h+13,dims,INK,8);                              /* length below the row */
                  else    s+=T(x-5,y+h/2+3,dims,INK,8,0,'var(--mono)','end'); }       /* length beside the column */
        return s;
      }
      function op(x,y,w,h,name,sub,col,dash,sub2){
        var d=dash?' stroke-dasharray="4 3"':' stroke-dasharray="5 3"';
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="9" fill="'+mix(col,7)+'" stroke="'+col+'" stroke-width="1.4"'+d+'/>';
        var mx=x+w/2,my=y+h/2;
        if(sub2){                                                   /* two-line spec: what the block contains */
          s+=T(mx,my-8,name,'var(--ink)',10.5,700,'var(--sans)');
          s+=T(mx,my+4,sub,INK,7.5);
          s+=T(mx,my+14,sub2,INK,7.5);
        }else{
          s+=T(mx,my-1,name,'var(--ink)',10.5,700,'var(--sans)'); if(sub) s+=T(mx,my+11,sub,INK,7.5);
        }
        return s;
      }
      function arr(pts,dir){
        var d='M'+pts[0][0]+' '+pts[0][1],i;
        for(i=1;i<pts.length;i++) d+=' L'+pts[i][0]+' '+pts[i][1];
        var e=pts[pts.length-1],hd;
        if(dir==='up') hd='M'+(e[0]-3.5)+' '+(e[1]+5)+' L'+e[0]+' '+e[1]+' L'+(e[0]+3.5)+' '+(e[1]+5);
        else if(dir==='down') hd='M'+(e[0]-3.5)+' '+(e[1]-5)+' L'+e[0]+' '+e[1]+' L'+(e[0]+3.5)+' '+(e[1]-5);
        else hd='M'+(e[0]-5)+' '+(e[1]-3.5)+' L'+e[0]+' '+e[1]+' L'+(e[0]-5)+' '+(e[1]+3.5);
        return '<path d="'+d+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.6"/><path d="'+hd+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>';
      }
      var P={
        seqI:{k:'vec',cx:40,cy:60,w:13,h:44,cells:5,col:'seq',name:'sequence',dims:'[L]',sec:'inputs'},
        esmc:{k:'op',cx:160,cy:60,w:100,h:48,col:'esmc',name:'ESMC-6B',sub:'80 layers · 6.35B',sub2:'frozen',sec:'esmc'},
        states:{k:'cube',cx:262,cy:60,w:38,h:34,dep:12,col:'esmc',name:'ESMC states',dims:'[L,81,2560]',sec:'esmc'},
        lmshim:{k:'op',cx:355,cy:60,w:86,h:48,col:'esmc',name:'LM shim',sub:'softmax over 81',sub2:'LN · → 256',sec:'esmc'},
        zlm:{k:'cube',cx:452,cy:60,w:44,h:40,dep:14,col:'pair',name:'z_lm',dims:'[L,L,256]',sec:'esmc'},
        atomI:{k:'mat',cx:40,cy:190,w:32,h:36,col:'atom',name:'atom feats',dims:'[N,389]',sec:'inputs'},
        inpemb:{k:'op',cx:240,cy:190,w:120,h:48,col:'seq',name:'Input embedder',sub:'atom SWA ×3 · + restype',sub2:'scatter-mean → 384',sec:'inputs'},
        finp:{k:'mat',cx:366,cy:190,w:30,h:38,col:'seq',name:'f_inputs',dims:'[L,451]',sec:'inputs'},
        zfeat:{k:'cube',cx:452,cy:190,w:44,h:40,dep:14,col:'pair',name:'z_feat',dims:'[L,L,256]',sec:'inputs'},
        msaI:{k:'cube',cx:44,cy:312,w:32,h:28,dep:10,col:'seq',name:'MSA·opt',dims:'[L,n_seq,128]',dash:1,sec:'inputs'},
        msaenc:{k:'op',cx:244,cy:312,w:116,h:48,col:'seq',name:'MSA encoder',sub:'OPM + pair-avg · ×4',sub2:'lifts MSA → pair',dash:1,sec:'parcae'},
        zmsa:{k:'cube',cx:452,cy:312,w:44,h:40,dep:14,col:'pair',name:'z_msa',dims:'[L,L,256]',sec:'parcae'},
        uin:{k:'op',cx:560,cy:190,w:108,h:48,col:'pair',name:'u inputs',sub:'z_feat + z_lm + z_msa',sub2:'re-injected ×T',sec:'parcae'},
        trunk:{k:'op',cx:774,cy:190,w:192,h:48,col:'pair',name:'parcae',sub:'z ← FoldingTrunk(Ā⊙z + B̄·LN(u))',sub2:'stable recurrence · ×T',sec:'parcae'},
        z0:{k:'op',cx:774,cy:286,w:150,h:40,col:'pair',name:'z_0 random init',sub:'trunc_norm noise · not z_feat',dash:1,sec:'parcae'},
        coda:{k:'op',cx:942,cy:190,w:104,h:48,col:'pair',name:'coda',sub:'linear',sub2:'+ 2× PairUpdateBlock',sec:'parcae'},
        zout:{k:'cube',cx:1048,cy:190,w:46,h:44,dep:16,col:'pair',name:'refined z',dims:'[L,L,256]',sec:'parcae'},
        disto:{k:'op',cx:1200,cy:90,w:132,h:42,col:'pair',name:'distogram head',sub:'Linear(z + zᵀ) · 64 bins',sec:'parcae'},
        diff:{k:'op',cx:1168,cy:190,w:96,h:48,col:'atom',name:'Diffusion',sub:'12× token DiT',sub2:'SWA + 3D RoPE',sec:'diffusion'},
        coords:{k:'mat',cx:1276,cy:190,w:36,h:44,col:'atom',name:'coords',dims:'[N,3]',sec:'diffusion'},
        conf:{k:'op',cx:1402,cy:190,w:108,h:48,col:'conf',name:'Confidence',sub:'4× PairUpdateBlock',sub2:'pLDDT · PAE · PDE',sec:'confidence'}
      };
      function hw(p){return (p.k==='cube')?(p.w+p.dep)/2:p.w/2;}
      function RX(p){return p.cx+hw(p)+6;}
      function LX(p){return p.cx-hw(p)-6;}
      function content(p){var col=C[p.col];
        if(p.k==='op') return op(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.sub,col,p.dash,p.sub2);
        if(p.k==='cube'){var vw=p.w+p.dep,x=p.cx-vw/2,y=p.cy-p.h/2;return cube(x,y,p.w,p.h,p.dep,p.name,p.dims,col,p.dash);}
        if(p.k==='vec') return vec(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.dims,col,p.cells,p.row);
        return mat(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.dims,col);
      }
      /* each dotted block deep-links to its own detail figure (falls back to its section) */
      var LINK={seqI:'fig-inputs',esmc:'fig-esmc',states:'fig-esmc',lmshim:'fig-lm',zlm:'fig-lm',
        atomI:'fig-inputs',inpemb:'fig-inputs',finp:'fig-inputs',zfeat:'fig-inputs',msaI:'fig-msa',msaenc:'fig-msa',zmsa:'fig-msa',
        uin:'fig-parcae',trunk:'fig-parcae',z0:'fig-parcae',coda:'fig-block',zout:'fig-parcae',disto:'figDistoSvg',
        diff:'fig-diffusion',coords:'fig-diffusion',conf:'fig-confidence'};
      function piece(p,link){return '<a href="#'+(link||SID[p.sec])+'">'+content(p)+'</a>';}
      var g='',yT=60,yM=190,yB=312,utop=P.uin.cy-P.uin.h/2,ubot=P.uin.cy+P.uin.h/2,uCx=P.uin.cx;
      /* language-model chain (top) → z_lm */
      g+=arr([[RX(P.seqI),yT],[LX(P.esmc),yT]]);
      g+=arr([[RX(P.esmc),yT],[LX(P.states),yT]]);
      g+=arr([[RX(P.states),yT],[LX(P.lmshim),yT]]);
      g+=arr([[RX(P.lmshim),yT],[LX(P.zlm),P.zlm.cy]]);         /* LM shim → z_lm cube */
      /* input-embedding chain (mid) → z_feat, plus restype from the sequence */
      g+=arr([[RX(P.atomI),yM],[LX(P.inpemb),yM]]);
      var jx=RX(P.seqI)+28;                                        /* the seq→ESMC line taps down to restype partway along */
      g+=arr([[jx,yT],[jx,178],[LX(P.inpemb),178]]);
      g+=arr([[RX(P.inpemb),yM],[LX(P.finp),yM]]);              /* Input embedder → f_inputs */
      g+=arr([[RX(P.finp),yM],[LX(P.zfeat),yM]]);               /* f_inputs → z_feat cube (outer sum) */
      /* MSA branch (bottom) → z_msa */
      g+=arr([[RX(P.msaI),yB],[LX(P.msaenc),yB]]);
      g+=arr([[RX(P.msaenc),yB],[LX(P.zmsa),yB]]);
      /* the three pair cubes converge into the per-loop inputs u */
      g+=arr([[RX(P.zlm),P.zlm.cy],[uCx,P.zlm.cy],[uCx,utop]]); /* z_lm down into u-inputs top */
      g+=arr([[RX(P.zfeat),yM],[LX(P.uin),yM]]);                /* z_feat into u-inputs left */
      g+=arr([[RX(P.zmsa),yB],[uCx,yB],[uCx,ubot]],'up');       /* z_msa up into u-inputs bottom */
      /* u inputs → parcae directly; the recurrent pair state is internal to parcae (per-loop u injection shown in the parcae section) */
      g+=arr([[RX(P.uin),yM],[LX(P.trunk),yM]]);
      /* z_0 is the recurrent state and starts random: it is NOT the featurization seed */
      g+=arr([[P.z0.cx,P.z0.cy-P.z0.h/2-6],[P.z0.cx,P.trunk.cy+P.trunk.h/2+6]],'up');
      g+=arr([[RX(P.trunk),yM],[LX(P.coda),yM]]);
      g+=arr([[RX(P.coda),yM],[LX(P.zout),yM]]);
      /* read out of z → diffusion → coords → confidence */
      g+=arr([[RX(P.zout),yM],[LX(P.diff),yM]]);
      g+=arr([[RX(P.diff),yM],[LX(P.coords),yM]]);
      g+=arr([[RX(P.coords),yM],[LX(P.conf),yM]]);
      /* both readouts of the finished pair state (the auxiliary distogram head, and the
         z that conditions the confidence head) fan out ABOVE the main lane, leaving the
         space below to the f_inputs track they used to tangle with */
      var zoutTX=P.zout.cx+hw(P.zout),                               /* top-right corner of the cube, */
          zoutTY=P.zout.cy-P.zout.h/2-Math.round(P.zout.dep*0.8);    /* clear of its name label */
      var yZ=134;
      g+=arr([[zoutTX,zoutTY],[zoutTX,P.disto.cy],[LX(P.disto),P.disto.cy]]);
      g+=arr([[zoutTX,zoutTY],[zoutTX,yZ],[P.conf.cx,yZ],[P.conf.cx,P.conf.cy-P.conf.h/2-6]],'down');
      /* f_inputs conditions both output heads: the static per-token track that replaces
         AF3's refined single representation */
      var yF=386,confB=P.conf.cy+P.conf.h/2+6,diffB=P.diff.cy+P.diff.h/2+6;
      var finpB=P.finp.cy+P.finp.h/2+16;
      g+=arr([[P.finp.cx,finpB],[P.finp.cx,yF],[P.diff.cx-16,yF],[P.diff.cx-16,diffB]],'up');
      g+=arr([[P.diff.cx-16,yF],[P.conf.cx-16,yF],[P.conf.cx-16,confB]],'up');
      for(var k in P) g+=piece(P[k],LINK[k]);
      /* flow captions */
      g+=T(jx-6,118,'restype',C.seq,8,700,'var(--mono)','end');
      g+=T(zoutTX+12,yZ+12,'z conditions the confidence head',C.pair,8,700,'var(--mono)','start');
      g+=T(P.finp.cx+14,yF-5,'f_inputs conditions both output heads (static, never refined)',C.seq,8,700,'var(--mono)','start');
      /* no section signposts: every block is itself a link to its detail figure */
      return '<svg viewBox="0 0 1500 404" role="img" aria-label="ESMFold2 pipeline. Three input chains feed the per-loop inputs u: the frozen ESMC language model produces z_lm, the atom-plus-sequence input embedder produces the static per-token features f_inputs and from them z_feat, and the optional MSA encoder produces z_msa, all summed in a u-inputs block. The inputs u feed the parcae module, a stable linear recurrence z ← FoldingTrunk(Ā⊙z + B̄·LN(u)) over 48 folding-trunk layers repeated T times; its recurrent state z_0 starts as random truncated normal noise rather than from the featurization, and a coda of one linear plus two pair layers finalizes the refined-z cube. Diffusion then reads coordinates out of the final z, a distogram head reads a distance histogram out of the same z as an auxiliary output, and the confidence head scores the structure. The refined z also conditions the confidence head, and f_inputs conditions both the diffusion and confidence heads as a static per-token track that is never refined.">'+g+'</svg>';
    }
    var f0=document.getElementById('fig0'); if(f0) f0.innerHTML=buildFig0();

    // Figure 3: featurization. Three families of input on the LEFT: atom
    // features, relative-position indices, and the covalent-bond graph, each
    // flow through one embedding box (the internals are deliberately not drawn),
    // and the three terms converge into the outer sum that seeds z_feat
    // (z_feat = OuterSum(f_inputs) + f_rel_pos + f_bond).
    function buildFig1(){
      var INK='var(--ink-faint)';
      var C={seq:'var(--seq)',esmc:'var(--esmc)',pair:'var(--pair)',atom:'var(--atom)',conf:'var(--conf)'};
      function T(x,y,t,cv,sz,w,fam,anchor){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||10)+'" fill="'+cv+'" text-anchor="'+(anchor||'middle')+'"'+(w?' font-weight="'+w+'"':'')+' font-family="'+(fam||'var(--mono)')+'">'+t+'</text>';}
      function cube(x,y,w,h,dep,name,dims,col){
        var dx=dep,dy=-Math.round(dep*0.8),s='';
        s+='<path d="M'+x+' '+y+' h'+w+' l'+dx+' '+dy+' h'+(-w)+' Z" fill="'+mix(col,20)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<path d="M'+(x+w)+' '+y+' l'+dx+' '+dy+' v'+h+' l'+(-dx)+' '+(-dy)+' Z" fill="'+mix(col,50)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<path d="M'+x+' '+y+' h'+w+' v'+h+' h'+(-w)+' Z" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+=T(x+w/2+dx/2,y+dy-7,name,col,10.5,700,'var(--sans)');
        if(dims) s+=T(x+w/2+dx/2,y+h+13,dims,INK,8);
        return s;
      }
      function mat(x,y,w,h,name,dims,col){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2" fill="'+mix(col,30)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<line x1="'+(x+w/2)+'" y1="'+y+'" x2="'+(x+w/2)+'" y2="'+(y+h)+'" stroke="'+col+'" stroke-opacity=".45"/>';
        s+='<line x1="'+x+'" y1="'+(y+h/2)+'" x2="'+(x+w)+'" y2="'+(y+h/2)+'" stroke="'+col+'" stroke-opacity=".45"/>';
        s+=T(x+w/2,y-7,name,col,10.5,700,'var(--sans)'); if(dims) s+=T(x+w/2,y+h+13,dims,INK,8);
        return s;
      }
      /* embed the shared opIcon() glyph(s) as positioned nested SVGs, so the SVG
         boxes keep the intuition-giving icons the old .mod/.tviz boxes carried */
      function glyph(spec,colKey,cx,cy,size){
        var ops=[].concat(spec),n=ops.length,gap=size+8,x0=cx-(n-1)*gap/2,out='';
        for(var i=0;i<n;i++){
          var raw=opIcon(ops[i],colKey),m=raw.match(/viewBox="0 0 ([0-9.]+) 40"/),vw=m?+m[1]:44,gh=size*40/vw,gx=x0+i*gap-size/2,gy=cy-gh/2;
          out+=raw.replace('<svg ','<svg x="'+gx+'" y="'+gy+'" width="'+size+'" height="'+gh+'" overflow="visible" ');
        }
        return out;
      }
      function op(x,y,w,h,name,lines,col,icon,colKey){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="10" fill="'+mix(col,7)+'" stroke="'+col+'" stroke-width="1.5" stroke-dasharray="5 3"/>';
        var mx=x+w/2,ny=y+21;
        s+=T(mx,ny,name,'var(--ink)',11.5,700,'var(--sans)');
        s+='<line x1="'+(x+14)+'" y1="'+(ny+7)+'" x2="'+(x+w-14)+'" y2="'+(ny+7)+'" stroke="'+col+'" stroke-opacity=".35"/>';
        (lines||[]).forEach(function(ln,i){ s+=T(mx,ny+21+i*12.5,ln,INK,8.5); });
        if(icon) s+=glyph(icon,colKey,mx,y+h-18,26);
        return s;
      }
      function arr(pts,dir){
        var d='M'+pts[0][0]+' '+pts[0][1],i;
        for(i=1;i<pts.length;i++) d+=' L'+pts[i][0]+' '+pts[i][1];
        var e=pts[pts.length-1],hd;
        if(dir==='up') hd='M'+(e[0]-3.5)+' '+(e[1]+5)+' L'+e[0]+' '+e[1]+' L'+(e[0]+3.5)+' '+(e[1]+5);
        else if(dir==='down') hd='M'+(e[0]-3.5)+' '+(e[1]-5)+' L'+e[0]+' '+e[1]+' L'+(e[0]+3.5)+' '+(e[1]-5);
        else hd='M'+(e[0]-5)+' '+(e[1]-3.5)+' L'+e[0]+' '+e[1]+' L'+(e[0]-5)+' '+(e[1]+3.5);
        return '<path d="'+d+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.6"/><path d="'+hd+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>';
      }
      var P={
        relIn:  {k:'mat', cx:74, cy:90,  w:52,h:32,        col:'seq', name:'indices',      dims:'[L, 4]'},
        relEmb: {k:'op',  cx:306,cy:90,  w:236,h:76,       col:'seq', name:'position embedding', lines:['how far apart along the chain,','same chain? same entity?']},
        frel:   {k:'cube',cx:524,cy:90,  w:44,h:36,dep:14, col:'pair',name:'f_rel_pos',   dims:'[L,L,256]'},
        restype:{k:'mat', cx:74, cy:176, w:52,h:30,        col:'seq', name:'restype',     dims:'[L, 33]'},
        atomIn: {k:'mat', cx:74, cy:262, w:52,h:36,        col:'atom',name:'atom feats',  dims:'[N,389]'},
        msaP:   {k:'mat', cx:74, cy:348, w:52,h:30,        col:'seq', name:'MSA profile', dims:'[L, 34]'},
        atomEmb:{k:'op',  cx:306,cy:262, w:236,h:92,       col:'atom',name:'atom embedding', lines:['sliding-window atom transformer,','pooled to one vector per token,','then ∥ restype ∥ MSA profile']},
        finp:   {k:'mat', cx:524,cy:262, w:40,h:40,        col:'seq', name:'f_inputs',    dims:'[L,451]'},
        osum:   {k:'op',  cx:712,cy:262, w:190,h:84,       col:'pair',name:'outer sum ⊕', lines:['z[i,j] = W₁·f_i + W₂·f_j','+ f_rel_pos','+ f_bond']},
        zfeat:  {k:'cube',cx:892,cy:262, w:48,h:42,dep:16, col:'pair',name:'z_feat',      dims:'[L,L,256]'},
        bondIn: {k:'mat', cx:74, cy:434, w:44,h:40,        col:'atom',name:'bond graph',  dims:'[L, L, 1]'},
        bondEmb:{k:'op',  cx:306,cy:434, w:236,h:76,       col:'atom',name:'bond embedding', lines:['which token pairs are held','together by a covalent bond']},
        fbond:  {k:'cube',cx:524,cy:434, w:44,h:36,dep:14, col:'pair',name:'f_bond',      dims:'[L,L,256]'}
      };
      function hw(p){return (p.k==='cube')?(p.w+p.dep)/2:p.w/2;}
      function RX(p){return p.cx+hw(p)+6;}
      function LX(p){return p.cx-hw(p)-6;}
      function content(p){var col=C[p.col];
        if(p.k==='op') return op(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.lines,col);
        if(p.k==='cube'){var vw=p.w+p.dep,x=p.cx-vw/2,y=p.cy-p.h/2;return cube(x,y,p.w,p.h,p.dep,p.name,p.dims,col);}
        return mat(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.dims,col);
      }
      var g='',yT=90,yM=262,yB=434,uCx=P.osum.cx,uTop=P.osum.cy-P.osum.h/2,uBot=P.osum.cy+P.osum.h/2;
      /* main lane (mid): atom features → atom embedding → f_inputs → outer sum → z_feat */
      g+=arr([[RX(P.atomIn),yM],[LX(P.atomEmb),yM]]);
      /* the atom embedding also ingests two per-token inputs from the left: residue type (above) and the MSA profile (below) */
      g+=arr([[RX(P.restype),P.restype.cy],[P.atomEmb.cx,P.restype.cy],[P.atomEmb.cx,P.atomEmb.cy-P.atomEmb.h/2]],'down');
      g+=arr([[RX(P.msaP),P.msaP.cy],[P.atomEmb.cx,P.msaP.cy],[P.atomEmb.cx,P.atomEmb.cy+P.atomEmb.h/2]],'up');
      g+=arr([[RX(P.atomEmb),yM],[LX(P.finp),yM]]);
      g+=arr([[RX(P.finp),yM],[LX(P.osum),yM]]);
      g+=arr([[RX(P.osum),yM],[LX(P.zfeat),yM]]);
      /* relative-position term (top): indices → position embedding → f_rel_pos → sum */
      g+=arr([[RX(P.relIn),yT],[LX(P.relEmb),yT]]);
      g+=arr([[RX(P.relEmb),yT],[LX(P.frel),yT]]);
      g+=arr([[RX(P.frel),yT],[uCx,yT],[uCx,uTop]],'down');
      /* token-bond term (bottom): bond graph → bond embedding → f_bond → sum */
      g+=arr([[RX(P.bondIn),yB],[LX(P.bondEmb),yB]]);
      g+=arr([[RX(P.bondEmb),yB],[LX(P.fbond),yB]]);
      g+=arr([[RX(P.fbond),yB],[uCx,yB],[uCx,uBot]],'up');
      for(var k in P) g+=content(P[k]);
      return '<svg viewBox="0 0 960 500" role="img" aria-label="Featurization and pair initialization. Three families of input on the left each pass through their own embedding. Relative-position indices [L,4] (how far apart two tokens are along the chain, and whether they belong to the same chain and the same entity) go through the position embedding, giving the pair term f_rel_pos [L,L,256]. Atom features [N,389] go through the atom embedding (a sliding-window transformer over atoms, pooled to one vector per token, then concatenated with residue type [L,33] and the optional MSA profile [L,34]), giving the per-token features f_inputs [L,451]. The covalent-bond graph [L,L,1] goes through the bond embedding, which marks the token pairs held together by a covalent bond, giving the pair term f_bond [L,L,256]. The three meet in an outer sum, z[i,j] = W1 f_i + W2 f_j + f_rel_pos + f_bond, that seeds the initial pair state z_feat [L,L,256].">'+g+'</svg>';
    }
    var f1=document.getElementById('fig1svg'); if(f1) f1.innerHTML=buildFig1();

    /* Shared SVG kit for the section-02 detail figures (Figures 5 and 6): the same
       dashed-box / cube / matrix / vector / arrow vocabulary as buildFig1, so the
       two figures read in the same visual language as Figure 3. */
    function figKit(){
      var INK='var(--ink-faint)';
      var C={seq:'var(--seq)',esmc:'var(--esmc)',pair:'var(--pair)',atom:'var(--atom)',conf:'var(--conf)'};
      function T(x,y,t,cv,sz,w,fam,anchor){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||10)+'" fill="'+cv+'" text-anchor="'+(anchor||'middle')+'"'+(w?' font-weight="'+w+'"':'')+' font-family="'+(fam||'var(--mono)')+'">'+t+'</text>';}
      function cube(x,y,w,h,dep,name,dims,col){
        var dx=dep,dy=-Math.round(dep*0.8),s='';
        s+='<path d="M'+x+' '+y+' h'+w+' l'+dx+' '+dy+' h'+(-w)+' Z" fill="'+mix(col,20)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<path d="M'+(x+w)+' '+y+' l'+dx+' '+dy+' v'+h+' l'+(-dx)+' '+(-dy)+' Z" fill="'+mix(col,50)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<path d="M'+x+' '+y+' h'+w+' v'+h+' h'+(-w)+' Z" fill="'+mix(col,34)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+=T(x+w/2+dx/2,y+dy-7,name,col,10.5,700,'var(--sans)');
        if(dims) s+=T(x+w/2+dx/2,y+h+13,dims,INK,8);
        return s;
      }
      function mat(x,y,w,h,name,dims,col){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2" fill="'+mix(col,30)+'" stroke="'+col+'" stroke-width="1.4"/>';
        s+='<line x1="'+(x+w/2)+'" y1="'+y+'" x2="'+(x+w/2)+'" y2="'+(y+h)+'" stroke="'+col+'" stroke-opacity=".45"/>';
        s+='<line x1="'+x+'" y1="'+(y+h/2)+'" x2="'+(x+w)+'" y2="'+(y+h/2)+'" stroke="'+col+'" stroke-opacity=".45"/>';
        s+=T(x+w/2,y-7,name,col,10.5,700,'var(--sans)'); if(dims) s+=T(x+w/2,y+h+13,dims,INK,8);
        return s;
      }
      function vec(x,y,w,h,name,dims,col,cells){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2" fill="'+mix(col,30)+'" stroke="'+col+'" stroke-width="1.4"/>';
        var n=cells||5,i;
        for(i=1;i<n;i++){var yy=y+h*i/n;s+='<line x1="'+x+'" y1="'+yy+'" x2="'+(x+w)+'" y2="'+yy+'" stroke="'+col+'" stroke-opacity=".45"/>';}
        s+=T(x+w/2,y-7,name,col,10.5,700,'var(--sans)');
        if(dims) s+=T(x-6,y+h/2+3,dims,INK,8,0,'var(--mono)','end');
        return s;
      }
      /* embed the shared opIcon() glyph(s) as positioned nested SVGs, so the SVG
         boxes keep the intuition-giving icons the old .mod/.tviz boxes carried */
      function glyph(spec,colKey,cx,cy,size){
        var ops=[].concat(spec),n=ops.length,gap=size+8,x0=cx-(n-1)*gap/2,out='';
        for(var i=0;i<n;i++){
          var raw=opIcon(ops[i],colKey),m=raw.match(/viewBox="0 0 ([0-9.]+) 40"/),vw=m?+m[1]:44,gh=size*40/vw,gx=x0+i*gap-size/2,gy=cy-gh/2;
          out+=raw.replace('<svg ','<svg x="'+gx+'" y="'+gy+'" width="'+size+'" height="'+gh+'" overflow="visible" ');
        }
        return out;
      }
      function op(x,y,w,h,name,lines,col,icon,colKey){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="10" fill="'+mix(col,7)+'" stroke="'+col+'" stroke-width="1.5" stroke-dasharray="5 3"/>';
        var mx=x+w/2,ny=y+21;
        s+=T(mx,ny,name,'var(--ink)',11.5,700,'var(--sans)');
        s+='<line x1="'+(x+14)+'" y1="'+(ny+7)+'" x2="'+(x+w-14)+'" y2="'+(ny+7)+'" stroke="'+col+'" stroke-opacity=".35"/>';
        (lines||[]).forEach(function(ln,i){ s+=T(mx,ny+21+i*12.5,ln,INK,8.5); });
        if(icon) s+=glyph(icon,colKey,mx,y+h-18,26);
        return s;
      }
      function arr(pts,dir){
        var d='M'+pts[0][0]+' '+pts[0][1],i;
        for(i=1;i<pts.length;i++) d+=' L'+pts[i][0]+' '+pts[i][1];
        var e=pts[pts.length-1],hd;
        if(dir==='up') hd='M'+(e[0]-3.5)+' '+(e[1]+5)+' L'+e[0]+' '+e[1]+' L'+(e[0]+3.5)+' '+(e[1]+5);
        else if(dir==='down') hd='M'+(e[0]-3.5)+' '+(e[1]-5)+' L'+e[0]+' '+e[1]+' L'+(e[0]+3.5)+' '+(e[1]-5);
        else hd='M'+(e[0]-5)+' '+(e[1]-3.5)+' L'+e[0]+' '+e[1]+' L'+(e[0]-5)+' '+(e[1]+3.5);
        return '<path d="'+d+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.6"/><path d="'+hd+'" fill="none" stroke="var(--ink-soft)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>';
      }
      /* a square matrix whose mass sits in a diagonal band: sliding-window attention,
         where each query only sees a fixed neighbourhood rather than every key */
      function band(x,y,w,h,name,dims,col){
        var s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="2" fill="'+mix(col,10)+'" stroke="'+col+'" stroke-width="1.4"/>';
        var bw=Math.round(Math.min(w,h)*0.34);                       /* band half-width in px */
        s+='<path d="M'+x+' '+(y+bw)+' L'+(x+bw)+' '+y+' L'+(x+w)+' '+(y+h-bw)+' L'+(x+w-bw)+' '+(y+h)+' Z" fill="'+mix(col,55)+'" stroke="none"/>';
        s+='<path d="M'+x+' '+y+' L'+(x+w)+' '+(y+h)+'" stroke="'+col+'" stroke-opacity=".5" stroke-width="1"/>';
        s+=T(x+w/2,y-7,name,col,10.5,700,'var(--sans)'); if(dims) s+=T(x+w/2,y+h+13,dims,INK,8);
        return s;
      }
      function hw(p){return (p.k==='cube')?(p.w+p.dep)/2:p.w/2;}
      function RX(p){return p.cx+hw(p)+6;}
      function LX(p){return p.cx-hw(p)-6;}
      function content(p){var col=C[p.col];
        if(p.k==='op') return op(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.lines,col,p.icon,p.col);
        if(p.k==='cube'){var vw=p.w+p.dep,x=p.cx-vw/2,y=p.cy-p.h/2;return cube(x,y,p.w,p.h,p.dep,p.name,p.dims,col);}
        if(p.k==='vec') return vec(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.dims,col,p.cells);
        if(p.k==='band') return band(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.dims,col);
        return mat(p.cx-p.w/2,p.cy-p.h/2,p.w,p.h,p.name,p.dims,col);
      }
      return {C:C,T:T,arr:arr,RX:RX,LX:LX,content:content};
    }

    /* Figure 5: inside frozen ESMC-6B. Token IDs → embedding → the repeated
       transformer layer → the kept 81-layer stack. Each output tensor is tracked. */
    function buildFigEsmc(){
      var K=figKit(),P={
        ids:   {k:'vec', cx:56, cy:98, w:15,h:50,cells:5,        col:'seq', name:'token IDs',        dims:'[L]'},
        embed: {k:'op',  cx:230,cy:98, w:158,h:100,              col:'esmc',name:'embed',            icon:'project',            lines:['h₀ = Embed[token id]','2560-d · bias-free']},
        emb:   {k:'mat', cx:398,cy:98, w:40,h:40,                col:'esmc',name:'token embeddings', dims:'[L, 2560]'},
        layer: {k:'op',  cx:636,cy:98, w:244,h:118,              col:'esmc',name:'transformer layer',icon:['attn','bottleneck'],lines:['h ← h + Attn(LN h)','h ← h + SwiGLU(LN h)','× 80 · frozen']},
        keep:  {k:'op',  cx:878,cy:98, w:158,h:100,              col:'esmc',name:'keep every layer', icon:'frozen-stack',       lines:['stack h₀, h₁, …, h₈₀','embedding + 80 = 81 states']},
        states:{k:'cube',cx:1074,cy:98,w:50,h:46,dep:16,         col:'esmc',name:'ESMC states',      dims:'[L, 81, 2560]'}
      };
      var g='';
      g+=K.arr([[K.RX(P.ids),P.ids.cy],[K.LX(P.embed),P.embed.cy]]);
      g+=K.arr([[K.RX(P.embed),P.embed.cy],[K.LX(P.emb),P.emb.cy]]);
      g+=K.arr([[K.RX(P.emb),P.emb.cy],[K.LX(P.layer),P.layer.cy]]);
      g+=K.arr([[K.RX(P.layer),P.layer.cy],[K.LX(P.keep),P.keep.cy]]);
      g+=K.arr([[K.RX(P.keep),P.keep.cy],[K.LX(P.states),P.states.cy]]);
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 1150 182" role="img" aria-label="Inside the frozen ESMC-6B language model. Token IDs of shape [L] pass through an embedding, h0 = Embed[token id], a bias-free lookup to 2560 dimensions, giving token embeddings [L, 2560]. These flow through one transformer layer, applied 80 times: a pre-LN residual update h ← h + Attn(LN h) with RoPE self-attention and QK-LayerNorm, then h ← h + SwiGLU(LN h), all bias-free and frozen. Every layer hidden state together with the embedding is kept (stacking h0, h1, up to h80, the embedding plus 80 layers equals 81 states), giving the ESMC states cube [L, 81, 2560].">'+g+'</svg>';
    }
    var fe=document.getElementById('figEsmcSvg'); if(fe) fe.innerHTML=buildFigEsmc();

    /* Figure 6: the LanguageModelShim. Distils the [L,81,2560] stack into one pair
       map z_lm, tracking LM single and the intermediate pair map, with the softmax
       mixture and outer-sum lift written out as equations. */
    function buildFigLm(){
      var K=figKit(),P={
        states:{k:'cube',cx:62, cy:100,w:46,h:40,dep:14,        col:'esmc',name:'ESMC states',    dims:'[L, 81, 2560]'},
        proj:  {k:'op',  cx:306,cy:100,w:258,h:118,            col:'esmc',name:'project & combine',icon:['project','combine'],lines:['LN · Linear · 2560 → 256','s = Σ_k softmax(w)_k · h_k','convex mix over 81 layers']},
        lms:   {k:'mat', cx:516,cy:100,w:34,h:44,               col:'esmc',name:'LM single',      dims:'[L, 256]'},
        s2p:   {k:'op',  cx:730,cy:100,w:200,h:102,            col:'pair',name:'Single → Pair',  icon:'outer',   lines:['z[i,j] = W₁·s_i + W₂·s_j','outer-sum lift → MLP']},
        pm:    {k:'cube',cx:924,cy:100,w:46,h:40,dep:14,        col:'pair',name:'pair map',       dims:'[L, L, 256]'},
        fold:  {k:'op',  cx:1118,cy:100,w:188,h:102,           col:'pair',name:'4× PairUpdateBlock',icon:'tri-out', lines:['the trunk block, 4 deep','LMEncoder · refines the map']},
        zlm:   {k:'cube',cx:1310,cy:100,w:52,h:48,dep:16,        col:'pair',name:'z_lm',           dims:'[L, L, 256]'}
      };
      var g='';
      g+=K.arr([[K.RX(P.states),P.states.cy],[K.LX(P.proj),P.proj.cy]]);
      g+=K.arr([[K.RX(P.proj),P.proj.cy],[K.LX(P.lms),P.lms.cy]]);
      g+=K.arr([[K.RX(P.lms),P.lms.cy],[K.LX(P.s2p),P.s2p.cy]]);
      g+=K.arr([[K.RX(P.s2p),P.s2p.cy],[K.LX(P.pm),P.pm.cy]]);
      g+=K.arr([[K.RX(P.pm),P.pm.cy],[K.LX(P.fold),P.fold.cy]]);
      g+=K.arr([[K.RX(P.fold),P.fold.cy],[K.LX(P.zlm),P.zlm.cy]]);
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 1366 184" role="img" aria-label="The language-model shim. The raw ESMC states cube [L, 81, 2560] is first projected and combined: LayerNorm and a Linear map 2560 to 256 per layer, then a length-81 learned weight vector through a softmax forms a convex combination s = sum over k of softmax(w)_k times h_k across the 81 layers, giving the LM single matrix [L, 256]. A Single-to-Pair module lifts it with the outer sum z[i,j] = W1 s_i + W2 s_j followed by an MLP, giving a pair map [L, L, 256], which four pair-folding LMEncoder layers refine into the pair map z_lm [L, L, 256].">'+g+'</svg>';
    }
    var fl=document.getElementById('figLmSvg'); if(fl) fl.innerHTML=buildFigLm();

    /* The MSA encoder (optional, demoted into its callout). Builds the pair update
       z_msa from the alignment via OuterProductMean, a pair-weighted MSA-stream
       update, and the two triangle multiplications: triangle attention dropped.
       Laid out over two rows so it stays legible at the narrow reading width. */
    function buildFigMsa(){
      var K=figKit(),P={
        msa:  {k:'cube',cx:52, cy:76, w:42,h:32,dep:12,       col:'seq', name:'MSA',             dims:'[L, n_seq, 128]'},
        opm:  {k:'op',  cx:240,cy:76, w:232,h:96,             col:'pair',name:'OuterProductMean',icon:'outer',  lines:['z[i,j] += mean_s(a_i ⊗ a_j)','MSA columns → pair']},
        pwa:  {k:'op',  cx:500,cy:76, w:200,h:96,             col:'pair',name:'pair-weighted avg',icon:'attn',  lines:['MSA ← pair-weighted mean','+ FFN on the MSA stream']},
        tri:  {k:'op',  cx:240,cy:212,w:232,h:104,            col:'pair',name:'4× PairUpdateBlock', icon:'tri-out',lines:['z ← z + TriMul_out(z)','z ← z + TriMul_in(z)','inside MSAEncoder · TriAttn dropped']},
        zmsa: {k:'cube',cx:520,cy:212,w:46,h:40,dep:14,       col:'pair',name:'z_msa',           dims:'[L, L, 256]'}
      };
      var g='';
      g+=K.arr([[K.RX(P.msa),P.msa.cy],[K.LX(P.opm),P.opm.cy]]);
      g+=K.arr([[K.RX(P.opm),P.opm.cy],[K.LX(P.pwa),P.pwa.cy]]);
      /* wrap from the end of the top row down into the second */
      g+=K.arr([[P.pwa.cx,P.pwa.cy+P.pwa.h/2+6],[P.pwa.cx,146],[P.tri.cx,146],[P.tri.cx,P.tri.cy-P.tri.h/2-6]],'down');
      g+=K.arr([[K.RX(P.tri),P.tri.cy],[K.LX(P.zmsa),P.zmsa.cy]]);
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 640 292" role="img" aria-label="The MSA encoder. The MSA cube [L, n_seq, 128] is turned into a pair update: OuterProductMean writes z[i,j] += mean over sequences s of the outer product a_i ⊗ a_j, carrying MSA columns into the pair representation; a pair-weighted average updates the MSA stream with an FFN; then the two triangle multiplications update the pair state, z ← z + TriMul_out(z) and z ← z + TriMul_in(z), over 4 layers with triangle attention dropped, giving the pair update z_msa [L, L, 256].">'+g+'</svg>';
    }
    var fm=document.getElementById('figMsaSvg'); if(fm) fm.innerHTML=buildFigMsa();

    /* Figure 8: inside one PairUpdateBlock. A residual spine of TriMul outgoing,
       TriMul incoming, PairTransition, with the identity skip drawn over each
       branch, the k-sum spelled out as a triangle underneath, and the deleted
       triangle attention left hanging off the spine as a ghost. */
    function buildFigBlock(){
      var K=figKit(),INK='var(--ink-faint)',PAIR='var(--pair)',yS=130,
      P={
        z:   {k:'cube',cx:60,  cy:yS,w:46,h:40,dep:14,      col:'pair',name:'z',               dims:'[L, L, 256]'},
        trio:{k:'op',  cx:278, cy:yS,w:216,h:118,           col:'pair',name:'TriMul outgoing', icon:'tri-out',   lines:['a, b, g ← gated Linear(LN z)','z[i,j] ← g ⊙ Σ_k a[i,k] ⊙ b[j,k]','edges leaving i and j · O(L³)']},
        trii:{k:'op',  cx:640, cy:yS,w:216,h:118,           col:'pair',name:'TriMul incoming', icon:'tri-in',    lines:['a, b, g ← gated Linear(LN z)','z[i,j] ← g ⊙ Σ_k a[k,i] ⊙ b[k,j]','edges entering i and j · O(L³)']},
        tran:{k:'op',  cx:980, cy:yS,w:216,h:118,           col:'pair',name:'PairTransition',  icon:'bottleneck',lines:['z ← Linear(SwiGLU(Linear(LN z)))','runs on each pair (i,j) alone','mixes channels, not positions']},
        zp:  {k:'cube',cx:1182,cy:yS,w:46,h:40,dep:14,      col:'pair',name:'z′',              dims:'[L, L, 256]'}
      };
      function T(x,y,t,cv,sz,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||8.5)+'" fill="'+cv+'" text-anchor="middle"'+(w?' font-weight="'+w+'"':'')+' font-family="var(--mono)">'+t+'</text>';}
      function plus(x){
        return '<g stroke="'+PAIR+'" stroke-width="1.6"><circle cx="'+x+'" cy="'+yS+'" r="9" fill="var(--surface)"/>'
             + '<line x1="'+(x-5)+'" y1="'+yS+'" x2="'+(x+5)+'" y2="'+yS+'"/><line x1="'+x+'" y1="'+(yS-5)+'" x2="'+x+'" y2="'+(yS+5)+'"/></g>';
      }
      function skip(x0,x1){
        return '<circle cx="'+x0+'" cy="'+yS+'" r="2.6" fill="'+PAIR+'"/>'
             + '<path d="M'+x0+' '+yS+' C'+x0+' 36 '+x1+' 36 '+x1+' '+(yS-14)+'" fill="none" stroke="'+PAIR+'" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#fbah)"/>';
      }
      /* the k-sum, drawn: edge (i,j) is rebuilt from the two edges that close the
         triangle through k: outgoing reads i→k and j→k, incoming reads k→i and k→j */
      function tri(cx,outgoing,cap){
        var ix=cx-56,jx=cx+56,ey=338,kx=cx,ky=268,s='';
        function edge(ax,ay,bx,by){
          var dx=bx-ax,dy=by-ay,n=Math.sqrt(dx*dx+dy*dy),ux=dx/n,uy=dy/n;
          return '<path d="M'+(ax+ux*8).toFixed(1)+' '+(ay+uy*8).toFixed(1)+' L'+(bx-ux*9).toFixed(1)+' '+(by-uy*9).toFixed(1)
               + '" fill="none" stroke="'+PAIR+'" stroke-width="1.5" stroke-opacity=".8" marker-end="url(#fbah)"/>';
        }
        function node(x,y,lbl,lx,ly){
          return '<circle cx="'+x+'" cy="'+y+'" r="3.6" fill="var(--surface)" stroke="'+PAIR+'" stroke-width="1.5"/>'+T(lx,ly,lbl,PAIR,10,700);
        }
        s+='<line x1="'+(ix+6)+'" y1="'+ey+'" x2="'+(jx-6)+'" y2="'+ey+'" stroke="'+PAIR+'" stroke-width="2.8"/>';
        s+=outgoing?edge(ix,ey,kx,ky)+edge(jx,ey,kx,ky):edge(kx,ky,ix,ey)+edge(kx,ky,jx,ey);
        s+=node(ix,ey,'i',ix-12,ey+4)+node(jx,ey,'j',jx+12,ey+4)+node(kx,ky,'k',kx,ky-9);
        return s+T(cx,ey+17,'z[i,j] updated',PAIR,8)+T(cx,ey+32,cap,INK,8.5);
      }
      var g='<defs><marker id="fbah" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="'+PAIR+'"/></marker></defs>';
      /* the residual spine */
      g+=K.arr([[K.RX(P.z),yS],[K.LX(P.trio),yS]]);
      g+=K.arr([[K.RX(P.trio),yS],[409,yS]]);
      g+=K.arr([[427,yS],[K.LX(P.trii),yS]]);
      g+=K.arr([[K.RX(P.trii),yS],[771,yS]]);
      g+=K.arr([[789,yS],[K.LX(P.tran),yS]]);
      g+=K.arr([[K.RX(P.tran),yS],[1111,yS]]);
      g+=K.arr([[1129,yS],[K.LX(P.zp),yS]]);
      g+=skip(130,418)+skip(478,780)+skip(830,1120);
      g+=plus(418)+plus(780)+plus(1120);
      g+=T(640,18,'identity skip: every branch is added back, so z is never overwritten',INK,9);
      /* what each branch costs */
      g+=T(278,207,'Dropout 0.25 on the branch',INK)+T(640,207,'Dropout 0.25 on the branch',INK);
      g+=T(980,207,'no dropout on this branch',INK);
      /* the same block, five places */
      g+=T(980,247,'one block, five places:',INK)+T(980,260,'48× trunk · 4× LM shim · 4× MSA',INK)+T(980,273,'2× coda · 4× confidence head',INK);
      /* the k-sum behind the two triangle updates */
      g+=tri(278,true,'outgoing: read i → k and j → k')+tri(640,false,'incoming: read k → i and k → j');
      g+=T(459,392,'every k closes a triangle on (i, j): i near k and k near j constrains i and j: the O(L³) sum, and the only step that mixes positions',INK,9);
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 1242 406" role="img" aria-label="Inside one PairUpdateBlock. The pair state z [L, L, 256] runs down a residual spine and leaves as z-prime [L, L, 256], the same shape. Three branches hang off it, each re-added at a plus node by an identity skip so z is never overwritten. First TriMul outgoing: gated linear projections a, b and gate g of the layer-normalized z, then z[i,j] receives g times the sum over k of a[i,k] elementwise b[j,k] (the edges leaving i and j), an O(L cubed) operation, with dropout 0.25 on the branch. Then TriMul incoming, identical but reading the edges entering i and j, the sum over k of a[k,i] elementwise b[k,j], also with dropout 0.25. Then the PairTransition, z ← Linear(SwiGLU(Linear(LN z))), which runs on each pair (i,j) alone and mixes channels rather than positions, with no dropout. Underneath, two triangles show the k-sum: for each pair (i,j), every third position k closes a triangle, outgoing reading the edges i to k and j to k, incoming reading k to i and k to j, so that i being near k and k being near j constrains i and j. This is the only step in the block that mixes positions. The same block stacks 48 times in the trunk, 4 times in the LM encoder, 4 times in the MSA encoder and twice in the coda.">'+g+'</svg>';
    }
    var fb=document.getElementById('figBlockSvg'); if(fb) fb.innerHTML=buildFigBlock();

    /* Figure 9: one diffusion step. Noisy atom coordinates are embedded, aggregated
       to tokens and refined by the pair-biased DiffusionTransformer, then decoded
       back to a per-atom coordinate update. */
    function buildFigDiff(){
      var K=figKit(),P={
        nx:  {k:'mat', cx:64, cy:102,w:38,h:44,               col:'atom',name:'noisy x',     dims:'[N_atom, 3]'},
        ae:  {k:'op',  cx:292,cy:102,w:208,h:100,             col:'atom',name:'AtomEncoder', icon:'window', lines:['SWA ×3 · window 128','3D RoPE · scatter-mean → tokens']},
        tok: {k:'mat', cx:472,cy:102,w:36,h:42,               col:'atom',name:'token repr',  dims:'[L, 768]'},
        dit: {k:'op',  cx:694,cy:102,w:204,h:100,             col:'pair',name:'Token DiT',   icon:'attn',   lines:['12 blocks · 16 heads · d 768','attn bias ← z · AdaLN(σ)']},
        ad:  {k:'op',  cx:924,cy:102,w:196,h:100,             col:'atom',name:'AtomDecoder', icon:'scatter',lines:['broadcast tokens → atoms','Linear → Δx']},
        xu:  {k:'mat', cx:1124,cy:102,w:40,h:44,              col:'atom',name:'x update',    dims:'[N_atom, 3]'},
        /* the three conditioning signals: this is where z is finally used */
        cz:  {k:'cube',cx:566,cy:246,w:34,h:30,dep:11,        col:'pair',name:'pair z',      dims:'[L,L,256]'},
        csig:{k:'op',  cx:706,cy:246,w:126,h:44,              col:'atom',name:'Fourier(σ)',  lines:['noise level']},
        cfin:{k:'mat', cx:850,cy:246,w:32,h:34,               col:'seq', name:'f_inputs',    dims:'[L,451]'}
      };
      var g='';
      g+=K.arr([[K.RX(P.nx),P.nx.cy],[K.LX(P.ae),P.ae.cy]]);
      g+=K.arr([[K.RX(P.ae),P.ae.cy],[K.LX(P.tok),P.tok.cy]]);
      g+=K.arr([[K.RX(P.tok),P.tok.cy],[K.LX(P.dit),P.dit.cy]]);
      g+=K.arr([[K.RX(P.dit),P.dit.cy],[K.LX(P.ad),P.ad.cy]]);
      g+=K.arr([[K.RX(P.ad),P.ad.cy],[K.LX(P.xu),P.xu.cy]]);
      /* conditioning enters the DiT from below: attention bias, AdaLN scale/shift, token features */
      var ditB=P.dit.cy+P.dit.h/2+6;
      g+=K.arr([[P.cz.cx,P.cz.cy-P.cz.h/2-18],[P.cz.cx,ditB+22],[P.dit.cx-58,ditB+22],[P.dit.cx-58,ditB]],'up');
      g+=K.arr([[P.csig.cx,P.csig.cy-P.csig.h/2-6],[P.csig.cx,ditB+22],[P.dit.cx,ditB+22],[P.dit.cx,ditB]],'up');
      g+=K.arr([[P.cfin.cx,P.cfin.cy-P.cfin.h/2-18],[P.cfin.cx,ditB+22],[P.dit.cx+58,ditB+22],[P.dit.cx+58,ditB]],'up');
      g+=K.T(P.dit.cx-62,ditB+16,'attn bias',K.C.pair,7.5,700,'var(--mono)','end');
      g+=K.T(P.dit.cx+62,ditB+16,'AdaLN + token cond.',K.C.atom,7.5,700,'var(--mono)','start');
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 1188 292" role="img" aria-label="One diffusion step. Noisy atom coordinates [N_atom, 3] are embedded by the AtomEncoder (three sliding-window-attention blocks with window 128 and 3D RoPE, scatter-mean pooled to tokens), giving a token representation [L, 768]. A 12-block Token DiffusionTransformer with 16 heads and d 768 refines it, its attention biased by the pair state z and conditioned on the noise level through AdaLN of sigma. An AtomDecoder broadcasts the tokens back to atoms and a Linear produces the coordinate update Δx, giving x update [N_atom, 3].">'+g+'</svg>';
    }
    var fd=document.getElementById('figDiffSvg'); if(fd) fd.innerHTML=buildFigDiff();

    /* Figure 10: the confidence head. Reads the pair state z and predicted coordinates,
       fans out to the four binned reliability heads. */
    function buildFigConf(){
      var K=figKit(),P={
        z:    {k:'cube',cx:72, cy:110,w:46,h:40,dep:14,        col:'pair',name:'pair z',        dims:'[L, L, 256]'},
        xp:   {k:'mat', cx:72, cy:232,w:40,h:40,               col:'atom',name:'x_pred',        dims:'[N_atom, 3]'},
        head: {k:'op',  cx:362,cy:170,w:224,h:118,             col:'conf',name:'ConfidenceHead',icon:'gauge',lines:['reads z + f_pair + distogram','4× PairUpdateBlock','binned distributions']},
        plddt:{k:'mat', cx:704,cy:48, w:40,h:40,               col:'conf',name:'pLDDT',         dims:'[L, 50]'},
        resl: {k:'mat', cx:704,cy:132,w:40,h:40,               col:'conf',name:'resolved',      dims:'[L, 2]'},
        pae:  {k:'cube',cx:712,cy:216,w:46,h:40,dep:14,        col:'conf',name:'PAE',           dims:'[L, L, 64]'},
        pde:  {k:'cube',cx:712,cy:300,w:46,h:40,dep:14,        col:'conf',name:'PDE',           dims:'[L, L, 64]'}
      };
      var g='',bus=582,hRX=K.RX(P.head),hLX=K.LX(P.head);
      g+=K.arr([[K.RX(P.z),P.z.cy],[222,P.z.cy],[222,150],[hLX,150]]);
      g+=K.arr([[K.RX(P.xp),P.xp.cy],[222,P.xp.cy],[222,190],[hLX,190]]);
      g+=K.arr([[hRX,P.head.cy],[bus,P.head.cy],[bus,P.plddt.cy],[K.LX(P.plddt),P.plddt.cy]]);
      g+=K.arr([[hRX,P.head.cy],[bus,P.head.cy],[bus,P.resl.cy],[K.LX(P.resl),P.resl.cy]]);
      g+=K.arr([[hRX,P.head.cy],[bus,P.head.cy],[bus,P.pae.cy],[K.LX(P.pae),P.pae.cy]]);
      g+=K.arr([[hRX,P.head.cy],[bus,P.head.cy],[bus,P.pde.cy],[K.LX(P.pde),P.pde.cy]]);
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 772 346" role="img" aria-label="The confidence head. Reading the pair state z [L, L, 256] and the predicted coordinates x_pred [N_atom, 3], a 4-layer pair network (which also sees a pair projection of the input features and an embedded distogram) emits binned distributions: per-residue pLDDT [L, 50] and an experimentally-resolved head [L, 2], and pairwise PAE [L, L, 64] and PDE [L, L, 64].">'+g+'</svg>';
    }
    var fc=document.getElementById('figConfSvg'); if(fc) fc.innerHTML=buildFigConf();

    /* The distogram head (aside in the folding-block section): a single linear readout
       of a distance histogram straight out of the finished pair state z. */
    function buildFigDisto(){
      var K=figKit(),P={
        z:    {k:'cube',cx:64, cy:74, w:42,h:38,dep:12,       col:'pair',name:'z',            dims:'[L, L, 256]'},
        head: {k:'op',  cx:300,cy:72, w:206,h:76,             col:'pair',name:'distogram head',lines:['logits = Linear(z + zᵀ)','64 bins · 2 to 22 Å · aux target']},
        disto:{k:'cube',cx:520,cy:72, w:44,h:40,dep:14,       col:'pair',name:'distogram',    dims:'[L, L, 64]'}
      };
      var g='';
      g+=K.arr([[K.RX(P.z),P.z.cy],[K.LX(P.head),P.head.cy]]);
      g+=K.arr([[K.RX(P.head),P.head.cy],[K.LX(P.disto),P.disto.cy]]);
      for(var k in P) g+=K.content(P[k]);
      return '<svg viewBox="0 0 570 124" width="100%" style="max-width:540px;display:block;margin:.4rem auto 0" role="img" aria-label="The distogram head: the finished pair state z [L, L, 256] passes through a single linear readout, logits = Linear(z + z-transpose), producing a distogram [L, L, 64]: a distribution over 64 distance bins from 2 to 22 Angstroms per pair, used as an auxiliary training target.">'+g+'</svg>';
    }
    var fdi=document.getElementById('figDistoSvg'); if(fdi) fdi.innerHTML=buildFigDisto();

    /* ---- mutual-information schematic: co-varying columns ⇒ contact map ≈ attention map ---- */
    function buildMI(){
      var G='var(--esmc)',Z='var(--pair)',INK='var(--ink-faint)',SOFT='var(--ink-soft)';
      function T(x,y,t,c,sz,w,anchor){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||9)+'" fill="'+c+'" text-anchor="'+(anchor||'middle')+'" font-family="var(--mono)"'+(w?' font-weight="'+w+'"':'')+'>'+t+'</text>';}
      function cell(x,y,s,fill,op){return '<rect x="'+x+'" y="'+y+'" width="'+s+'" height="'+s+'" rx="1.5" fill="'+fill+'"'+(op!=null?' fill-opacity="'+op+'"':'')+'/>';}
      var g='',r,c;
      /* Panel A: an alignment (5 rows x 6 cols); columns i=1 and j=4 co-vary */
      var ax=14,ay=56,cs=13,pit=16,rows=5,cols=6,hi=[1,4];
      for(r=0;r<rows;r++)for(c=0;c<cols;c++){
        var on=hi.indexOf(c)>=0;
        g+=cell(ax+c*pit,ay+r*pit,cs,on?G:INK,on?(0.32+0.26*((r+c)%2)):0.16);
      }
      var xi=ax+hi[0]*pit+cs/2,xj=ax+hi[1]*pit+cs/2,ya=ay-4;
      g+='<path d="M'+xi+' '+ya+' Q'+((xi+xj)/2)+' '+(ya-24)+' '+xj+' '+ya+'" fill="none" stroke="'+G+'" stroke-width="1.4"/>';
      g+=T((xi+xj)/2,ya-28,'mutual information',G,8,700);
      g+=T(xi,ay+rows*pit+9,'i',G,8.5,700); g+=T(xj,ay+rows*pit+9,'j',G,8.5,700);
      g+=T(ax+(cols*pit)/2-pit/2,ay+rows*pit+22,'co-varying columns',INK,8);
      /* contact map / attention map: identical L x L grids, same off-diagonal (i,j) cell */
      function grid(ox,oy,col,title){
        var s='',rr,cc,n=7,cz=12,pz=13,side=n*pz-pz+cz;
        s+='<rect x="'+ox+'" y="'+oy+'" width="'+side+'" height="'+side+'" rx="2" fill="none" stroke="'+col+'" stroke-opacity=".5" stroke-width="1"/>';
        for(rr=0;rr<n;rr++)for(cc=0;cc<n;cc++){
          if(rr===cc) s+=cell(ox+cc*pz,oy+rr*pz,cz,col,0.26);
          if((rr===1&&cc===4)||(rr===4&&cc===1)) s+=cell(ox+cc*pz,oy+rr*pz,cz,col,0.85);
        }
        /* name the axes at the marked cell: the figure's one inferential step is
           reading alignment columns i, j as the row and column of an L×L map */
        s+=T(ox-7,oy+pz+cz/2+3,'i',col,8,700,'end');
        s+=T(ox+4*pz+cz/2,oy+side+11,'j',col,8,700);
        return s+T(ox+side/2,oy-7,title,col,9,700);
      }
      var abx=ax+cols*pit+4;
      var bx=abx+42; g+=grid(bx,50,Z,'contact map');
      var apx=bx+90; g+=T(apx+13,99,'≈',SOFT,18,700)+T(apx+13,114,'mirror',INK,7);
      var cx=apx+30; g+=grid(cx,50,G,'ESMC attention');
      /* stream: the covariation of columns i,j surfaces as the (i,j) contact cell */
      var gpz=13,gcz=12,bcx=bx+4*gpz+gcz/2,bcy=50+1*gpz+gcz/2,s0x=(xi+xj)/2,s0y=ya-18;
      g+='<path d="M'+s0x+' '+s0y+' C'+(s0x+72)+' '+(s0y-16)+','+(bcx-48)+' '+(bcy-30)+','+bcx+' '+bcy+'" fill="none" stroke="'+G+'" stroke-width="1.4" stroke-opacity=".75"/>';
      g+='<path d="M'+(bcx-9)+' '+(bcy-3)+' L'+bcx+' '+bcy+' L'+(bcx-3)+' '+(bcy-9)+'" fill="none" stroke="'+G+'" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>';
      return '<svg viewBox="0 0 '+(cx+102)+' 162" aria-hidden="true">'+g+'</svg>';
    }
    var mi=document.getElementById('mifig'); if(mi) mi.innerHTML=buildMI();

    /* ---- outer-sum inset: one single vector, read down the rows and across the
       columns, lifts to an L×L pair map whose (i,j) cell is W₁sᵢ + W₂sⱼ ---- */
    function buildOuterSum(){
      var S='var(--seq)',Z='var(--pair)',INK='var(--ink-faint)',SOFT='var(--ink-soft)';
      function T(x,y,t,c,sz,w,anchor,fam){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||9)+'" fill="'+c+'" text-anchor="'+(anchor||'middle')+'" font-family="'+(fam||'var(--mono)')+'"'+(w?' font-weight="'+w+'"':'')+'>'+t+'</text>';}
      function cell(x,y,s,fill,op){return '<rect x="'+x+'" y="'+y+'" width="'+s+'" height="'+s+'" rx="1.5" fill="'+fill+'"'+(op!=null?' fill-opacity="'+op+'"':'')+'/>';}
      function arr(x1,y1,x2,y2){
        var hd;
        if(x1===x2){var dy=y2>y1?1:-1;hd='M'+(x2-3.5)+' '+(y2-dy*5)+' L'+x2+' '+y2+' L'+(x2+3.5)+' '+(y2-dy*5);}
        else{var dx=x2>x1?1:-1;hd='M'+(x2-dx*5)+' '+(y2-3.5)+' L'+x2+' '+y2+' L'+(x2-dx*5)+' '+(y2+3.5);}
        return '<path d="M'+x1+' '+y1+' L'+x2+' '+y2+'" stroke="'+SOFT+'" stroke-width="1.5" fill="none"/><path d="'+hd+'" stroke="'+SOFT+'" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>';
      }
      var g='',r,c,n=7,cz=16,pz=18,i=1,j=4,gx=214,gy=64,side=n*pz-pz+cz,lx=gx-40,ty=gy-40;
      /* the single s, read down the rows (one cell per row → sᵢ) */
      for(r=0;r<n;r++) g+=cell(lx,gy+r*pz,cz,r===i?S:INK,r===i?0.9:0.16);
      /* the SAME single s, read across the columns (one cell per column → sⱼ) */
      for(c=0;c<n;c++) g+=cell(gx+c*pz,ty,cz,c===j?S:INK,c===j?0.9:0.16);
      /* the L×L pair map: faint diagonal + row-i and col-j bands, lit (i,j) cell */
      g+='<rect x="'+gx+'" y="'+gy+'" width="'+side+'" height="'+side+'" rx="2" fill="none" stroke="'+Z+'" stroke-opacity=".5" stroke-width="1"/>';
      for(r=0;r<n;r++)for(c=0;c<n;c++){
        var op=0; if(r===c)op=0.12; if(r===i)op=Math.max(op,0.2); if(c===j)op=Math.max(op,0.2);
        if(op>0) g+=cell(gx+c*pz,gy+r*pz,cz,Z,op);
      }
      g+=cell(gx+j*pz,gy+i*pz,cz,Z,0.92);
      g+=T(gx+j*pz+cz/2,gy+i*pz+cz/2+4,'+','var(--surface)',12,700);
      /* projection arrows converging on the lit cell */
      g+=arr(lx+cz,gy+i*pz+cz/2,gx-3,gy+i*pz+cz/2);
      g+=arr(gx+j*pz+cz/2,ty+cz,gx+j*pz+cz/2,gy-3);
      g+=T((lx+cz+gx)/2,gy+i*pz-3,'W₁',Z,8,700);
      g+=T(gx+j*pz+cz/2+8,(ty+cz+gy)/2+3,'W₂',Z,8,700,'start');
      /* labels */
      g+=T(gx+side/2,ty-8,'the single  s  ·  read across  j',S,8.5,700,'middle','var(--sans)');
      g+=T(lx+cz/2,gy+n*pz+7,'…and down  i',S,8.5,700);
      g+=T(gx-6,gy+i*pz+cz/2+3,'i',SOFT,8,'end');
      g+=T(gx+j*pz+cz/2,gy-7,'j',SOFT,8);
      g+=T(gx+side/2,gy+side+16,'pair map  z',Z,9.5,700,'middle','var(--sans)');
      g+=T(gx+side/2,gy+side+28,'[L, L, 256]',INK,8);
      /* the formula, to the right of the map */
      var fx=gx+side+22,fy=gy+side/2;
      g+=T(fx,fy-7,'z[ i, j ]  =','var(--ink)',11,700,'start','var(--sans)');
      g+=T(fx+4,fy+11,'W₁ sᵢ  +  W₂ sⱼ',Z,10.5,700,'start','var(--sans)');
      g+=T(fx+4,fy+27,'[L, 256]  →  [L, L, 256]',INK,7.5,0,'start');
      return '<svg viewBox="0 0 500 232" aria-hidden="true">'+g+'</svg>';
    }
    var os=document.getElementById('outersum'); if(os) os.innerHTML=buildOuterSum();

    /* ---- Figure 2: flag horizontal overflow so the edge fade + hint appear ---- */
    var f0box=document.getElementById('fig0-overview'), f0wrap=f0box&&f0box.closest('.fig0scroll');
    function onF0(){ if(f0box&&f0wrap) f0wrap.classList.toggle('overflow', f0box.scrollWidth-f0box.clientWidth>4); }
    onF0(); window.addEventListener('resize',onF0,{passive:true});
  })();


  /* Shared floating tooltip for glossary terms (.term) and reference marks (sup.ref a).
     Any element carrying data-note opts in. Hover/focus on desktop; tap toggles on touch. */
  (function(){
    var tip=document.createElement('div');tip.id='tip';tip.setAttribute('role','tooltip');document.body.appendChild(tip);
    var cur=null,hideT=null;
    function place(el){
      tip.style.maxWidth=Math.min(352,window.innerWidth-24)+'px';
      tip.classList.add('show');
      var r=el.getBoundingClientRect(),t=tip.getBoundingClientRect(),
          sx=window.scrollX,sy=window.scrollY;
      var left=r.left+sx+r.width/2-t.width/2;
      left=Math.max(sx+12,Math.min(left,sx+window.innerWidth-t.width-12));
      var top=r.top+sy-t.height-9;
      if(r.top-t.height-9<0) top=r.bottom+sy+9;   /* flip below when no room above */
      tip.style.left=left+'px';tip.style.top=top+'px';
    }
    function show(el){
      var note=el.getAttribute('data-note');if(!note)return;
      if(hideT){clearTimeout(hideT);hideT=null;}
      var isRef=el.closest('sup.ref')!==null;
      tip.textContent='';
      var lb=document.createElement('span');lb.className='tw';
      lb.textContent=isRef?'Reference':'Definition';
      tip.appendChild(lb);tip.appendChild(document.createTextNode(note));
      if(cur&&cur!==el) cur.removeAttribute('aria-describedby');
      cur=el;el.setAttribute('aria-describedby','tip');place(el);
    }
    function hide(){hideT=setTimeout(function(){tip.classList.remove('show');if(cur)cur.removeAttribute('aria-describedby');cur=null;},80);}
    document.querySelectorAll('[data-note]').forEach(function(el){
      el.addEventListener('mouseenter',function(){show(el);});
      el.addEventListener('mouseleave',hide);
      el.addEventListener('focus',function(){show(el);});
      el.addEventListener('blur',hide);
    });
    tip.addEventListener('mouseenter',function(){if(hideT){clearTimeout(hideT);hideT=null;}});
    tip.addEventListener('mouseleave',hide);
    /* touch: tapping a term toggles its note; references keep their normal link jump */
    document.addEventListener('click',function(e){
      var t=e.target.closest('.term');
      if(t){ if(cur===t){tip.classList.remove('show');t.removeAttribute('aria-describedby');cur=null;} else show(t); return; }
      if(!e.target.closest('#tip')){ tip.classList.remove('show');if(cur)cur.removeAttribute('aria-describedby');cur=null; }
    });
    window.addEventListener('scroll',function(){if(cur){tip.classList.remove('show');cur.removeAttribute('aria-describedby');cur=null;}},{passive:true});
  })();
