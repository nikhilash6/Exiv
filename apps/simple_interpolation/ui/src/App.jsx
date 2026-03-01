import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const SimpleInterpolationUI = ({ appName = 'Simple Interpolation' }) => {
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState('');
  const [error, setError] = useState('');

  const [initialImage, setInitialImage] = useState('');
  const [finalImage, setFinalImage] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const [positive, setPositive] = useState('smooth transition');
  const [negative, setNegative] = useState('bad quality, static, deformed');

  const [height, setHeight] = useState(720);
  const [width, setWidth] = useState(720);
  const [videoLength, setVideoLength] = useState(5);

  const [finalVideo, setFinalVideo] = useState('');

  const initialImgRef = useRef(null);
  const finalImgRef = useRef(null);

  const actions = window.useTaskActions ? window.useTaskActions() : null;
  const addTask = actions ? actions.addTask : null;

  const { addToast } = window.useToast ? window.useToast() : { addToast: (msg) => alert(msg.message || msg) };

  const resolveMediaUrl = (path) => {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    if (path.startsWith('/api/outputs')) return path;
    return `/api/outputs/${path}`;
  };

  const runTask = async () => {
    if (!initialImage || !finalImage) {
      addToast({ message: 'Please upload both initial and final images', type: 'error' });
      return;
    }

    setError('');
    try {
      const res = await fetch('/api/apps/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          app_name: appName,
          params: {
            initial_image: initialImage,
            final_image: finalImage,
            positive,
            negative,
            seed: -1,
            steps: 20,
            cfg: 6.0,
            width,
            height,
            frame_count: Math.ceil(videoLength * 16),
          },
        }),
      });

      const data = await res.json();
      setTaskId(data.task_id);
      setFinalVideo('');

      if (addTask) {
        addTask({
          id: data.task_id,
          name: appName,
          status: 'queued',
          progress: 0,
        });
      }
    } catch (err) {
      setError(err.message || 'Failed to start task');
    }
  };

  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/status/${taskId}`);
        const data = await res.json();

        setTaskStatus(data.status);

        if (data.status === 'completed') {
          const output = data.output['1'];
          setTaskId(null);
          setError('');
          setFinalVideo(output);
        } else if (data.status === 'failed') {
          const errorMsg = data.data?.err_message || 'Task failed';
          setError(errorMsg);
          setTaskId(null);
        }
      } catch (e) {
        console.error(e);
        setError('Error fetching task status');
        setTaskId(null);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId]);

  const handleFileUpload = async (file, setPathCallback) => {
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'success') {
        setPathCallback(data.file_path);
      }
    } catch (err) {
      console.error('Upload error', err);
    } finally {
      setIsUploading(false);
    }
  };

  const statusClass =
    taskStatus === 'failed' ? 'failed' : taskId ? 'running' : finalVideo ? 'done' : 'idle';

  if (isUploading) {
    return (
      <div className="awa-loading-screen">
        <div className="awa-spinner" />
      </div>
    );
  }

  return (
    <div className="awa-page">
      <div className="awa-bg" />
      <div className="awa-shell">
        <header className="awa-header">
          <div>
            <h1>Simple Interpolation</h1>
            <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: 'var(--awa-muted)' }}>create a simple video by interpolating between the start and end frames</p>
          </div>
          <div className={`awa-run-status ${statusClass}`}>
            <span>{taskId ? 'Running' : 'Status'}</span>
            <strong>{taskId ? taskStatus || 'queued' : finalVideo ? 'ready' : 'idle'}</strong>
          </div>
        </header>

        {error && (
          <div className="awa-alert" style={{ animationDelay: '120ms' }}>
            {error}
          </div>
        )}

        <main className="awa-grid">
          <section className="awa-card full-width">
            <div className="awa-card-head">
              <h2>1. Upload Keyframes</h2>
            </div>
            
            <div className="awa-init-layout" style={{ gap: '24px' }}>
              <div className="awa-seg-main" style={{ flex: '1 1 50%', maxWidth: '50%', paddingRight: '12px', borderRight: '1px solid var(--awa-card-border)' }}>
                <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Initial Frame</h3>
                <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '16px' }}>The starting image of the video.</p>
                <input
                  ref={initialImgRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileUpload(e.target.files[0], setInitialImage)}
                />
                <div 
                  className="awa-placeholder" 
                  style={{ width: '100%', height: '300px', cursor: 'pointer', overflow: 'hidden' }}
                  onClick={() => initialImgRef.current.click()}
                >
                  {initialImage ? (
                    <img src={resolveMediaUrl(initialImage)} alt="Initial Frame" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  ) : (
                    <span>Click to upload initial frame</span>
                  )}
                </div>
              </div>

              <div className="awa-init-sidebar" style={{ flex: '1 1 50%', maxWidth: '50%', paddingLeft: '12px' }}>
                <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Final Frame</h3>
                <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '16px' }}>The ending image of the video.</p>
                <input
                  ref={finalImgRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileUpload(e.target.files[0], setFinalImage)}
                />
                <div 
                  className="awa-placeholder" 
                  style={{ width: '100%', height: '300px', cursor: 'pointer', overflow: 'hidden' }}
                  onClick={() => finalImgRef.current.click()}
                >
                  {finalImage ? (
                    <img src={resolveMediaUrl(finalImage)} alt="Final Frame" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  ) : (
                    <span>Click to upload final frame</span>
                  )}
                </div>
              </div>
            </div>
          </section>

          <section className="awa-card full-width" style={{ animationDelay: '260ms' }}>
            <div className="awa-card-head">
              <h2>2. Interpolate</h2>
            </div>
            <div className="awa-init-layout">
              <div className="awa-init-sidebar">
                <label className="awa-label" htmlFor="positivePrompt">Prompt</label>
                <textarea
                  id="positivePrompt"
                  className="awa-textarea"
                  value={positive}
                  onChange={(e) => setPositive(e.target.value)}
                  rows={3}
                />
                
                <div className="awa-row">
                  <div className="awa-col">
                    <label className="awa-label">Width</label>
                    <input
                      type="number"
                      className="awa-input"
                      value={width}
                      onChange={(e) => setWidth(parseInt(e.target.value))}
                    />
                  </div>
                  <div className="awa-col">
                    <label className="awa-label">Height</label>
                    <input
                      type="number"
                      className="awa-input"
                      value={height}
                      onChange={(e) => setHeight(parseInt(e.target.value))}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: '14px' }}>
                  <label className="awa-label">
                    Video Length: {videoLength} secs
                  </label>
                  <input
                    type="range"
                    className="awa-input"
                    style={{ padding: 0 }}
                    min={2}
                    max={10}
                    step={1}
                    value={videoLength}
                    onChange={(e) => setVideoLength(parseFloat(e.target.value))}
                  />
                </div>

                <button
                  className="awa-btn primary"
                  type="button"
                  onClick={runTask}
                  disabled={!!taskId || isUploading}
                >
                  Generate Interpolation
                </button>
              </div>
              <div className="awa-seg-main">
                <div className="awa-media-panel">
                  <h3>Final Output</h3>
                  {finalVideo ? (
                    <video className="awa-final-video" controls src={resolveMediaUrl(finalVideo)} autoPlay loop />
                  ) : (
                    <div className="awa-placeholder">Final interpolated video will appear here</div>
                  )}
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};

export default SimpleInterpolationUI;
