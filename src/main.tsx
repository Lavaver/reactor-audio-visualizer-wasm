// No StrictMode: its development double-mount would create and tear down two
// AudioContexts and two animation loops, which is noise for an audio page.
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

const container = document.getElementById('root');
if (!container) throw new Error('缺少 #root 挂载点');

createRoot(container).render(<App />);
