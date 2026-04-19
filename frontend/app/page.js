'use client';
import { useState, useRef, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000';

export default function Home() {
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState('');
  const [loading, setLoading]     = useState(false);
  const [temp, setTemp]           = useState(0.8);
  const bottomRef                 = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    const userMsg    = { role: 'user', content: text };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setLoading(true);
    try {
      const res  = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: messages, temperature: temp }),
      });
      const data = await res.json();
      setMessages([...newHistory, { role: 'assistant', content: data.content }]);
    } catch {
      setMessages([...newHistory, { role: 'assistant', content: '[Error — is the backend running?]' }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }

  return (
    <div style={s.root}>
      <header style={s.header}>
        <div style={s.logo}>
          <span style={s.logoMark}>◈</span>
          <span style={s.logoText}>CUSTOM<span style={s.logoAccent}>GPT</span></span>
        </div>
        <div style={s.controls}>
          <span style={s.label}>TEMP</span>
          <input type="range" min="0.1" max="1.5" step="0.05" value={temp}
            onChange={e => setTemp(parseFloat(e.target.value))} style={s.slider} />
          <span style={s.tempVal}>{temp.toFixed(2)}</span>
          <span style={s.msgCount}>{messages.length} msgs</span>
        </div>
      </header>

      <main style={s.main}>
        {messages.length === 0 && (
          <div style={s.empty}>
            <div style={s.emptyGlyph}>◈</div>
            <div style={s.emptyTitle}>Your model. Zero dependencies.</div>
            <div style={s.emptySub}>No OpenAI. No Anthropic. Just your weights.</div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={m.role === 'user' ? s.userRow : s.aiRow}>
            <div style={m.role === 'user' ? s.userBubble : s.aiBubble}>
              <span style={m.role === 'user' ? s.userLabel : s.aiLabel}>
                {m.role === 'user' ? 'YOU' : 'GPT'}
              </span>
              <p style={s.msgText}>{m.content}</p>
            </div>
          </div>
        ))}
        {loading && (
          <div style={s.aiRow}>
            <div style={s.aiBubble}>
              <span style={s.aiLabel}>GPT</span>
              <div style={s.dots}>
                {[0, 150, 300].map(d => (
                  <span key={d} style={{ ...s.dot, animationDelay: `${d}ms` }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer style={s.footer}>
        <div style={s.inputRow}>
          <textarea value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey} placeholder="Message… (Enter to send)"
            rows={1} style={s.textarea} />
          <button onClick={send} disabled={loading || !input.trim()} style={s.sendBtn}>
            {loading ? '…' : '↑'}
          </button>
        </div>
      </footer>

      <style>{`
        @keyframes bounce {
          0%,80%,100% { transform:translateY(0); opacity:0.3; }
          40% { transform:translateY(-5px); opacity:1; }
        }
        textarea { resize:none; }
        input[type=range] { accent-color:#7c6af7; cursor:pointer; }
        button:hover:not(:disabled) { background:#5c4de0 !important; }
        button:disabled { opacity:0.35; cursor:not-allowed; }
      `}</style>
    </div>
  );
}

const s = {
  root:       { display:'flex', flexDirection:'column', height:'100vh', maxWidth:'780px', margin:'0 auto' },
  header:     { padding:'14px 20px', borderBottom:'1px solid #1c1c30', display:'flex', alignItems:'center', justifyContent:'space-between', background:'rgba(8,8,16,0.92)', backdropFilter:'blur(12px)', position:'sticky', top:0, zIndex:10 },
  logo:       { display:'flex', alignItems:'center', gap:'8px' },
  logoMark:   { color:'#7c6af7', fontSize:'18px' },
  logoText:   { fontFamily:"'IBM Plex Mono',monospace", fontSize:'13px', fontWeight:600, letterSpacing:'3px' },
  logoAccent: { color:'#7c6af7' },
  controls:   { display:'flex', alignItems:'center', gap:'10px' },
  label:      { fontFamily:"'IBM Plex Mono',monospace", fontSize:'9px', color:'#4a4a6a', letterSpacing:'2px' },
  slider:     { width:'72px' },
  tempVal:    { fontFamily:"'IBM Plex Mono',monospace", fontSize:'11px', color:'#7c6af7', minWidth:'34px' },
  msgCount:   { fontFamily:"'IBM Plex Mono',monospace", fontSize:'9px', color:'#4a4a6a', letterSpacing:'1px' },
  main:       { flex:1, overflowY:'auto', padding:'20px', display:'flex', flexDirection:'column', gap:'14px' },
  empty:      { flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:'10px', opacity:0.35 },
  emptyGlyph: { fontSize:'36px', color:'#7c6af7' },
  emptyTitle: { fontFamily:"'IBM Plex Mono',monospace", fontSize:'14px', letterSpacing:'1px' },
  emptySub:   { fontSize:'12px', color:'#4a4a6a' },
  userRow:    { display:'flex', justifyContent:'flex-end' },
  aiRow:      { display:'flex', justifyContent:'flex-start' },
  userBubble: { background:'#141428', border:'1px solid #2a2a50', borderRadius:'16px 16px 4px 16px', padding:'11px 15px', maxWidth:'72%' },
  aiBubble:   { background:'#0d0d1e', border:'1px solid #1c1c30', borderRadius:'16px 16px 16px 4px', padding:'11px 15px', maxWidth:'72%' },
  userLabel:  { fontFamily:"'IBM Plex Mono',monospace", fontSize:'8px', letterSpacing:'2px', color:'#7c6af7', display:'block', marginBottom:'5px' },
  aiLabel:    { fontFamily:"'IBM Plex Mono',monospace", fontSize:'8px', letterSpacing:'2px', color:'#4ecca3', display:'block', marginBottom:'5px' },
  msgText:    { fontSize:'14px', lineHeight:'1.65', color:'#e2e2f0', whiteSpace:'pre-wrap' },
  dots:       { display:'flex', gap:'4px', alignItems:'center', height:'18px' },
  dot:        { width:'5px', height:'5px', borderRadius:'50%', background:'#4ecca3', display:'inline-block', animation:'bounce 1.2s infinite' },
  footer:     { padding:'14px 20px', borderTop:'1px solid #1c1c30', background:'rgba(8,8,16,0.95)' },
  inputRow:   { display:'flex', gap:'8px', alignItems:'flex-end' },
  textarea:   { flex:1, background:'#0f0f1c', border:'1px solid #1c1c30', borderRadius:'12px', padding:'11px 15px', color:'#e2e2f0', fontFamily:"'DM Sans',sans-serif", fontSize:'14px', outline:'none', lineHeight:'1.5', minHeight:'42px', maxHeight:'180px' },
  sendBtn:    { width:'42px', height:'42px', borderRadius:'12px', background:'#7c6af7', border:'none', color:'#fff', fontSize:'17px', cursor:'pointer', display:'flex', alignItems:'center', justifyContent:'center', fontWeight:700, transition:'background 0.15s', flexShrink:0 },
};
