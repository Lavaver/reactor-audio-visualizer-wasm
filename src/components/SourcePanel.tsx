import { useCallback, useEffect, useRef, useState } from 'react';
import type { EngineSnapshot, ReactorEngine } from '../reactor/engine';

interface SourcePanelProps {
  engine: ReactorEngine;
  snapshot: EngineSnapshot;
}

export function SourcePanel({ engine, snapshot }: SourcePanelProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [seekPreview, setSeekPreview] = useState(0);
  const [device, setDevice] = useState('');
  const draggingRef = useRef(false);
  const previewRef = useRef(0);

  // Default to the first loopback device the browser offered, if any.
  useEffect(() => {
    if (!snapshot.devices?.length) {
      setDevice('');
      return;
    }
    setDevice(current =>
      snapshot.devices?.some(option => option.id === current) ? current : snapshot.devices![0].id,
    );
  }, [snapshot.devices]);

  const commitSeek = useCallback(
    (value: number) => {
      draggingRef.current = false;
      setDragging(false);
      engine.seekTo(value);
    },
    [engine],
  );

  // While dragging, the snapshot keeps ticking; only the release commits, so the
  // buffer is not restarted on every pointer move. Listening on `window` also
  // catches a release that happens outside the slider.
  useEffect(() => {
    if (!dragging) return;
    const finish = () => commitSeek(previewRef.current);
    window.addEventListener('pointerup', finish);
    window.addEventListener('pointercancel', finish);
    return () => {
      window.removeEventListener('pointerup', finish);
      window.removeEventListener('pointercancel', finish);
    };
  }, [dragging, commitSeek]);

  const showDeviceBox = snapshot.devices !== null || snapshot.deviceHint !== null;
  const seekValue = dragging ? seekPreview : snapshot.seekPercent;

  return (
    <section className="source">
      <label className="caption">01 / 音频输入</label>
      <div className="source-tabs">
        <button
          id="demo"
          title="通过一个简单的例子为你演示它的玩法"
          className={snapshot.mode === 'demo' ? 'selected' : undefined}
          onClick={() => void engine.selectMode('demo')}
        >
          演示信号
        </button>
        <button
          id="mic"
          title="使用麦克风录制的音频"
          className={snapshot.mode === 'mic' ? 'selected' : undefined}
          onClick={() => void engine.selectMode('mic')}
        >
          麦克风
        </button>
        <button
          id="system"
          title="共享标签页或屏幕的音频"
          className={snapshot.mode === 'system' ? 'selected' : undefined}
          onClick={() => void engine.selectMode('system')}
        >
          机内声音
        </button>
      </div>

      {showDeviceBox && (
        <>
          <p id="systemnote" className="source-note">
            {snapshot.deviceHint ?? 'Firefox 不能直接共享系统声音；请选择系统回环输入设备，或改用 Chrome / Edge。'}
          </p>
          {snapshot.devices && snapshot.devices.length > 0 && (
            <div id="sysdevicebox" className="device-box">
              <label htmlFor="sysdevices">系统回环设备</label>
              <select id="sysdevices" value={device} onChange={event => setDevice(event.target.value)}>
                {snapshot.devices.map(option => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button
                id="sysdevicestart"
                className="device-start"
                disabled={!device}
                onClick={() => void engine.startWithDevice(device)}
              >
                开始监听
              </button>
              <small>首次使用需要授权音频输入（Firefox 会请求麦克风权限）。</small>
            </div>
          )}
        </>
      )}

      <button id="upload" className="upload" onClick={() => fileRef.current?.click()}>
        <span>＋</span>
        <strong>导入音频文件</strong>
        <small>MP3 / WAV / M4A / FLAC</small>
      </button>
      <input
        id="file"
        type="file"
        accept="audio/*"
        hidden
        ref={fileRef}
        onChange={event => {
          const file = event.target.files?.[0];
          if (file) void engine.loadFile(file);
          event.target.value = '';
        }}
      />

      <div className="track">
        <span className="track-icon">≋</span>
        <div>
          <strong id="trackname">{snapshot.trackName}</strong>
          <small id="tracksub">{snapshot.trackSub}</small>
        </div>
        <button
          id="play"
          aria-label={snapshot.playing ? '暂停' : '播放'}
          onClick={() => engine.toggle()}
        >
          {snapshot.playing ? 'Ⅱ' : '▶'}
        </button>
      </div>

      <div className="timeline">
        <input
          id="seek"
          aria-label="播放进度"
          type="range"
          min={0}
          max={100}
          step={0.1}
          value={seekValue}
          disabled={!snapshot.seekEnabled}
          onPointerDown={() => {
            draggingRef.current = true;
            previewRef.current = snapshot.seekPercent;
            setSeekPreview(snapshot.seekPercent);
            setDragging(true);
          }}
          onChange={event => {
            const value = Number(event.target.value);
            previewRef.current = value;
            setSeekPreview(value);
            // Keyboard adjustment (no pointer drag) applies immediately.
            if (!draggingRef.current) commitSeek(value);
          }}
          onKeyUp={event => commitSeek(Number(event.currentTarget.value))}
        />
        <div>
          <span id="elapsed">{snapshot.elapsedLabel}</span>
          <span id="duration">{snapshot.durationLabel}</span>
        </div>
      </div>
    </section>
  );
}
