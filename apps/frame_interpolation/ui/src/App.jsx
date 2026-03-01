import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const SimpleInterpolationUI = ({ appName = 'Simple Interpolation' }) => {
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState('');
  const [error, setError] = useState('');

  const [keyframes, setKeyframes] = useState(['', '']);
  const [prompts, setPrompts] = useState(['smooth transition']);
  const [durations, setDurations] = useState([5]);
  const [uploadingIndex, setUploadingIndex] = useState(null);

  const [height, setHeight] = useState(512);
  const [width, setWidth] = useState(512);
  const [finalVideo, setFinalVideo] = useState('');

  const keyframeInputRefs = useRef([]);

  const actions = window.useTaskActions ? window.useTaskActions() : null;
  const addTask = actions ? actions.addTask : null;

  const { addToast } = window.useToast ? window.useToast() : { addToast: (msg) => alert(msg.message || msg) };

  const resolveMediaUrl = (path) => {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    if (path.startsWith('/api/outputs')) return path;
    return `/api/outputs/${path}`;
  };

  const addKeyframe = () => {
    setKeyframes([...keyframes, '']);
    setPrompts([...prompts, 'smooth transition']);
    setDurations([...durations, 5]);
  };

  const removeKeyframe = (index) => {
    if (keyframes.length <= 2) return;
    const newKeyframes = [...keyframes];
    newKeyframes.splice(index, 1);
    setKeyframes(newKeyframes);

    const newPrompts = [...prompts];
    // If we remove a keyframe, we remove the prompt that was after it, 
    // unless it was the last keyframe, then we remove the prompt before it.
    if (index < prompts.length) {
      newPrompts.splice(index, 1);
    } else {
      newPrompts.splice(index - 1, 1);
    }
    setPrompts(newPrompts);

    const newDurations = [...durations];
    if (index < durations.length) {
      newDurations.splice(index, 1);
    } else {
      newDurations.splice(index - 1, 1);
    }
    setDurations(newDurations);
  };

  const updateKeyframe = (index, val) => {
    const newKeyframes = [...keyframes];
    newKeyframes[index] = val;
    setKeyframes(newKeyframes);
  };

  const updatePrompt = (index, val) => {
    const newPrompts = [...prompts];
    newPrompts[index] = val;
    setPrompts(newPrompts);
  };

  const updateDuration = (index, val) => {
    const newDurations = [...durations];
    newDurations[index] = parseInt(val);
    setDurations(newDurations);
  };

  const runTask = async () => {
    if (keyframes.some(kf => !kf)) {
      addToast({ message: 'Please upload all keyframes', type: 'error' });
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
            keyframes,
            prompts,
            durations,
            seed: -1,
            steps: 20,
            cfg: 6.0,
            width,
            height,
          },
        }),
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);
      
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

  const handleFileUpload = async (file, index) => {
    if (!file) return;

    setUploadingIndex(index);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'success') {
        updateKeyframe(index, data.file_path);
      }
    } catch (err) {
      console.error('Upload error', err);
    } finally {
      setUploadingIndex(null);
    }
  };

  const statusClass =
    taskStatus === 'failed' ? 'failed' : taskId ? 'running' : finalVideo ? 'done' : 'idle';

  return (
    <div className="awa-page">
      <div className="awa-bg" />
      <div className="awa-shell">
        <header className="awa-header">
          <div>
            <h1>Frame Interpolation</h1>
            <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: 'var(--awa-muted)' }}>Create complex video transitions by interpolating through multiple keyframes</p>
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
              <h2>1. Keyframe Sequence</h2>
            </div>
            
            <div className="keyframes-scroll-container">
              <div className="keyframes-sequence">
                {keyframes.map((kf, index) => (
                  <React.Fragment key={index}>
                    <div className="keyframe-block">
                      <div className="keyframe-label">
                        <span>Frame {index + 1}</span>
                        {keyframes.length > 2 && (
                          <button className="remove-kf-btn" onClick={() => removeKeyframe(index)}>×</button>
                        )}
                      </div>
                      <div 
                        className="awa-placeholder kf-preview" 
                        onClick={() => keyframeInputRefs.current[index].click()}
                      >
                        {uploadingIndex === index ? (
                          <div className="awa-spinner" />
                        ) : kf ? (
                          <img src={resolveMediaUrl(kf)} alt={`Frame ${index + 1}`} />
                        ) : (
                          <span>Click to upload</span>
                        )}
                      </div>
                      <input
                        ref={el => keyframeInputRefs.current[index] = el}
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={(e) => { handleFileUpload(e.target.files[0], index); e.target.value = null; }}
                      />
                    </div>

                    {index < keyframes.length - 1 && (
                      <div className="segment-settings">
                        <div className="segment-arrow">→</div>
                        <div className="segment-inputs">
                          <textarea
                            className="awa-textarea segment-prompt"
                            placeholder="Prompt for this transition..."
                            value={prompts[index]}
                            onChange={(e) => updatePrompt(index, e.target.value)}
                            rows={2}
                          />
                          <div className="duration-selector">
                            <label>Duration:</label>
                            <select 
                              value={durations[index]} 
                              onChange={(e) => updateDuration(index, e.target.value)}
                              className="awa-input"
                              style={{ marginBottom: 0, padding: '4px 8px', width: 'auto' }}
                            >
                              <option value={3}>3s</option>
                              <option value={4}>4s</option>
                              <option value={5}>5s</option>
                            </select>
                          </div>
                        </div>
                      </div>
                    )}
                  </React.Fragment>
                ))}
                
                <button className="add-keyframe-card" onClick={addKeyframe}>
                  <div className="plus-icon">+</div>
                  <span>Add Frame</span>
                </button>
              </div>
            </div>
          </section>

          <section className="awa-card full-width">
            <div className="awa-card-head">
              <h2>2. Global Settings & Output</h2>
            </div>
            <div className="awa-init-layout">
              <div className="awa-init-sidebar">
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

                <button
                  className="awa-btn primary full-width"
                  type="button"
                  onClick={runTask}
                  disabled={!!taskId || uploadingIndex !== null}
                  style={{ marginTop: '10px', width: '100%' }}
                >
                  Generate Video Sequence
                </button>
                <p style={{ marginTop: '10px', fontSize: '12px', color: 'var(--awa-muted)', textAlign: 'center' }}>
                  Note: This uses Wan VACE 14B, which can take up to 6 minutes to generate a 5-second video on an RTX A5000
                </p>
              </div>
              
              <div className="awa-seg-main">
                <div className="awa-media-panel">
                  <h3>Final Interpolated Video</h3>
                  {finalVideo ? (
                    <video className="awa-final-video" controls src={resolveMediaUrl(finalVideo)} autoPlay loop />
                  ) : (
                    <div className="awa-placeholder" style={{ height: '300px' }}>
                      Final sequence will appear here after generation
                    </div>
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
