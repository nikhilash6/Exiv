import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const AdvancedWanAnimateUI = ({ appName = 'Character Replace' }) => {
  const [step, setStep] = useState(0);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState('');
  const [error, setError] = useState('');
  const [isAutoChaining, setIsAutoChaining] = useState(false);

  const [inputVideo, setInputVideo] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [firstFrame, setFirstFrame] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const [points, setPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [previewMask, setPreviewMask] = useState('');

  const [fgVideo, setFgVideo] = useState('');
  const [maskVideo, setMaskVideo] = useState('');
  const [poseVideo, setPoseVideo] = useState('');
  const [faceVideo, setFaceVideo] = useState('');
  const [finalVideo, setFinalVideo] = useState('');

  const [positive, setPositive] = useState('a girl talking');
  const [negative, setNegative] = useState('bad quality');
  const [referenceImage, setReferenceImage] = useState('');

  const [height, setHeight] = useState(640);
  const [width, setWidth] = useState(640);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoLength, setVideoLength] = useState(5);

  const imgRef = useRef(null);
  const videoInputRef = useRef(null);
  const refImageInputRef = useRef(null);

  const actions = window.useTaskActions ? window.useTaskActions() : null;
  const addTask = actions ? actions.addTask : null;
  
  const { addToast } = window.useToast ? window.useToast() : { addToast: () => alert('Please create mask and upload the reference image') };

  const resolveMediaUrl = (path) => {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    if (path.startsWith('/api/outputs')) return path;
    return `/api/outputs/${path}`;
  };

  const runTask = async (mode, params) => {
    setError('');
    const res = await fetch('/api/apps/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        app_name: appName,
        params: { app_mode: mode, ...params },
      }),
    });

    const data = await res.json();
    setTaskId(data.task_id);

    if (addTask) {
      addTask({
        id: data.task_id,
        name: `Character Replace - ${mode}`,
        status: 'queued',
        progress: 0,
      });
    }
    return data.task_id;
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
          setTaskId(null); // Clear taskId first to prevent multiple triggers
          setError('');

          if (step === 0) {
            setFirstFrame(output.first_frame);
            setSessionId(output.session_id);
            setStep(1);
          } else if (step === 1) {
            setPreviewMask(output.preview);
          } else if (step === 2) {
            setFgVideo(output.fg_video);
            setMaskVideo(output.mask_video);
            if (isAutoChaining) {
              setStep(3);
              runTask('3_pose', { input_video: inputVideo });
            }
          } else if (step === 3) {
            setPoseVideo(output.pose_video);
            setFaceVideo(output.face_video);
            if (isAutoChaining) {
              setStep(4);
              runTask('4_animate', {
                bg_video: output.fg_video || fgVideo,
                mask_video: output.mask_video || maskVideo,
                pose_video: output.pose_video,
                face_video: output.face_video,
                reference_image: referenceImage,
                positive,
                negative: '',
                seed: -1,
                steps: 4,
                cfg: 1.0,
                width,
                height,
                frame_count: Math.ceil(videoLength * 16),
              });
            }
          } else if (step === 4) {
            setFinalVideo(output.final_video);
            setStep(5);
            setIsAutoChaining(false);
          }
        } else if (data.status === 'failed') {
          const errorMsg = data.data?.err_message || 'Task failed';
          setError(errorMsg);
          setTaskId(null);
          setIsAutoChaining(false);
        }
      } catch (e) {
        console.error(e);
        setError('Error fetching task status');
        setTaskId(null);
        setIsAutoChaining(false);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId, step, isAutoChaining, inputVideo, sessionId, positive, referenceImage, width, height, videoLength, fgVideo, maskVideo]);

  const handleRenderFinal = () => {
    if (!referenceImage) {
      if (window.useToast) {
        addToast({ message: 'Please upload the reference image', type: 'error' });
      } else {
        alert('Please upload the reference image');
      }
      return;
    }
    setIsAutoChaining(true);
    if (inputVideo && previewMask) {
      setStep(2);
      runTask('2_matte', { input_video: inputVideo, session_id: sessionId });
    } else {
      setStep(4);
      runTask('4_animate', {
        bg_video: '',
        mask_video: '',
        pose_video: '',
        face_video: '',
        reference_image: referenceImage,
        positive,
        negative: '',
        seed: -1,
        steps: 4,
        cfg: 1.0,
        width,
        height,
        frame_count: Math.ceil(videoLength * 16),
      });
    }
  };

  const handleFileUpload = async (file, setPathCallback, autoInit = false) => {
    if (!file) return;

    if (file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src);
        const duration = Math.floor(video.duration);
        setVideoDuration(duration);
        setVideoLength(Math.min(duration, 5));
      };
      video.src = URL.createObjectURL(file);
    }

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
        if (autoInit) {
          setStep(0);
          runTask('0_init', { input_video: data.file_path });
        }
      }
    } catch (err) {
      console.error('Upload error', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleImageClick = (e) => {
    if (!imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();

    const scaleX = imgRef.current.naturalWidth / rect.width;
    const scaleY = imgRef.current.naturalHeight / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    const isRightClick = e.type === 'contextmenu';
    if (isRightClick) e.preventDefault();

    setPoints((prev) => [...prev, [x, y]]);
    setLabels((prev) => [...prev, isRightClick ? 0 : 1]);
  };

  const clearPoints = () => {
    setPoints([]);
    setLabels([]);
    setPreviewMask('');
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
            <h1>Character Replace</h1>
            <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: 'var(--awa-muted)' }}>replace a character from the input video to the one provided by you</p>
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
              <h2>1. Initialize & Segment</h2>
              <span className={previewMask ? 'pill done' : 'pill'}>
                {previewMask ? 'Mask Ready' : firstFrame ? 'Awaiting Mask' : 'Ready'}
              </span>
            </div>
            
            <div className="awa-init-layout" style={{ gap: '24px' }}>
              {/* Left Column: Load Video & Mask Generation */}
              <div className="awa-seg-main" style={{ flex: '1 1 50%', maxWidth: '50%', paddingRight: '12px', borderRight: '1px solid var(--awa-card-border)' }}>
                <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Load Source Video</h3>
                <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '4px' }}>Pick source video and initialize session.</p>
                <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '4px' }}>* Videos greater than 5 secs are preferred.</p>
                <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '16px' }}>* If input video is not provided, then the ref image will be animated.</p>
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileUpload(e.target.files[0], setInputVideo, true)}
                />
                <button
                  className="awa-btn primary"
                  type="button"
                  onClick={() => videoInputRef.current.click()}
                  disabled={!!taskId || isUploading}
                >
                  Load Video
                </button>
                <p className="awa-path" style={{ marginTop: '12px', marginBottom: '40px' }}>{inputVideo ? `Loaded: ${inputVideo}` : 'No video selected'}</p>

                <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Mask Generation</h3>
                <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '16px' }}>Click to mark a point, mask will be generated for this marked segment</p>
                <div className="awa-media-row" style={{ gridTemplateColumns: '1fr', margin: '0 0 20px 0' }}>
                  <div className="awa-media-panel">
                    <h3>{previewMask ? 'Mask Preview (Click to refine)' : 'Frame'}</h3>
                    {firstFrame ? (
                      <div className="awa-image-wrap">
                        <img
                          ref={imgRef}
                          src={resolveMediaUrl(previewMask || firstFrame)}
                          onClick={handleImageClick}
                          onContextMenu={handleImageClick}
                          alt="preview"
                        />
                        {points.map((p, i) => (
                          <span
                            key={`${p[0]}-${p[1]}-${i}`}
                            className={`awa-point ${labels[i] === 1 ? 'fg' : 'bg'}`}
                            style={{
                              left: `${(p[0] / (imgRef.current?.naturalWidth || 1)) * 100}%`,
                              top: `${(p[1] / (imgRef.current?.naturalHeight || 1)) * 100}%`,
                            }}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="awa-placeholder" style={{ height: '300px' }}>Initialize video to see frame</div>
                    )}
                  </div>
                </div>
                <div className="awa-actions">
                  <button
                    className="awa-btn primary"
                    type="button"
                    onClick={() =>
                      runTask('1_segment', {
                        input_video: inputVideo,
                        session_id: sessionId,
                        points: JSON.stringify(points),
                        labels: JSON.stringify(labels),
                      })
                    }
                    disabled={!!taskId || points.length === 0 || !sessionId}
                  >
                    Generate Mask
                  </button>
                  <button className="awa-btn ghost" type="button" onClick={clearPoints}>
                    Clear Mask
                  </button>
                </div>
              </div>

              {/* Right Column: Select New Character */}
              <div className="awa-init-sidebar" style={{ flex: '1 1 50%', maxWidth: '50%', paddingLeft: '12px' }}>
                <div style={{ marginBottom: '40px' }}>
                  <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Select the character to replace</h3>
                  <p style={{ fontSize: '12px', color: 'var(--awa-muted)', marginBottom: '16px' }}>it should have a white background</p>
                  <input
                    ref={refImageInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFileUpload(e.target.files[0], setReferenceImage)}
                  />
                  <div 
                    className="awa-placeholder" 
                    style={{ width: '100%', height: '300px', cursor: 'pointer', overflow: 'hidden' }}
                    onClick={() => refImageInputRef.current.click()}
                  >
                    {referenceImage ? (
                      <img src={resolveMediaUrl(referenceImage)} alt="reference" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    ) : (
                      <span>Click to select reference image</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="awa-card full-width" style={{ animationDelay: '260ms' }}>
            <div className="awa-card-head">
              <h2>2. Animate</h2>
              <span className={finalVideo ? 'pill done' : 'pill'}>{finalVideo ? 'Rendered' : taskId && isAutoChaining ? 'Generating...' : 'Ready to Run'}</span>
            </div>
            <div className="awa-init-layout">
              <div className="awa-init-sidebar">
                <p>Configure prompts and reference image, then render output.</p>
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
                    min={videoDuration > 0 ? Math.min(videoDuration, 5) : 5}
                    max={videoDuration > 0 ? Math.max(videoDuration, 10) : 10}
                    step={1}
                    value={videoLength}
                    onChange={(e) => setVideoLength(parseFloat(e.target.value))}
                  />
                </div>

                <button
                  className="awa-btn primary"
                  type="button"
                  onClick={handleRenderFinal}
                  disabled={!!taskId || isUploading}
                >
                  Render Final Animation
                </button>
              </div>
              <div className="awa-seg-main">
                <div className="awa-media-panel">
                  <h3>Final Output</h3>
                  {finalVideo ? (
                    <video className="awa-final-video" controls src={resolveMediaUrl(finalVideo)} />
                  ) : (
                    <div className="awa-placeholder">Final rendered video will appear here</div>
                  )}
                </div>

                {/* Intermediate results shown below */}
                {(fgVideo || poseVideo) && (
                  <div className="awa-intermediate-results" style={{ marginTop: '32px', borderTop: '1px solid var(--awa-card-border)', paddingTop: '24px' }}>
                    <h2 style={{ fontSize: '14px', marginBottom: '16px', color: 'var(--awa-muted)' }}>Intermediate Generation Steps</h2>
                    <div className="awa-media-row">
                      <div className="awa-media-panel">
                        <h3>Foreground Matting</h3>
                        {fgVideo ? <video controls src={resolveMediaUrl(fgVideo)} /> : <div className="awa-placeholder">Processing...</div>}
                      </div>
                      <div className="awa-media-panel">
                        <h3>Pose Estimation</h3>
                        {poseVideo ? <video controls src={resolveMediaUrl(poseVideo)} /> : <div className="awa-placeholder">Processing...</div>}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};

export default AdvancedWanAnimateUI;
