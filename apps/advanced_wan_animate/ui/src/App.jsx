import React, { useState, useEffect, useRef } from 'react';

const AdvancedWanAnimateUI = ({ appName = "Advanced Wan Animate" }) => {
  const [step, setStep] = useState(0);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState("");
  const [error, setError] = useState("");
  
  // Data state
  const [inputVideo, setInputVideo] = useState("dialogue.mp4");
  const [sessionId, setSessionId] = useState("");
  const [firstFrame, setFirstFrame] = useState("");
  const [isRemote, setIsRemote] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // Segmentation state
  const [points, setPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [previewMask, setPreviewMask] = useState("");
  
  // Paths state
  const [fgVideo, setFgVideo] = useState("");
  const [maskVideo, setMaskVideo] = useState("");
  const [poseVideo, setPoseVideo] = useState("");
  const [faceVideo, setFaceVideo] = useState("");
  const [finalVideo, setFinalVideo] = useState("");
  
  // Animation params
  const [positive, setPositive] = useState("a girl talking");
  const [negative, setNegative] = useState("bad quality");
  const [referenceImage, setReferenceImage] = useState("ref_image.png");
  
  const imgRef = useRef(null);

  const actions = window.useTaskActions ? window.useTaskActions() : null;
  const addTask = actions ? actions.addTask : null;

  // Use a helper to resolve media URLs just in case
  const resolveMediaUrl = (path) => {
    if (!path) return "";
    if (path.startsWith("http")) return path;
    if (path.startsWith("/api/outputs")) return path;
    // Exiv backend returns paths relative to the output folder.
    // The server exposes these at /api/outputs/{filename}
    return `/api/outputs/${path}`;
  };

  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/status/${taskId}`);
        const data = await res.json();
        
        setTaskStatus(data.status);

        if (data.status === 'completed') {
          const output = data.output["1"];
          if (step === 0) {
            setFirstFrame(output.first_frame);
            setSessionId(output.session_id);
            setStep(1);
          } else if (step === 1) {
            setPreviewMask(output.preview);
          } else if (step === 2) {
            setFgVideo(output.fg_video);
            setMaskVideo(output.mask_video);
            setStep(3);
          } else if (step === 3) {
            setPoseVideo(output.pose_video);
            setFaceVideo(output.face_video);
            setStep(4);
          } else if (step === 4) {
            setFinalVideo(output.final_video);
            setStep(5);
          }
          setTaskId(null);
          setError("");
        } else if (data.status === 'failed') {
          const errorMsg = data.data?.err_message || "Task Failed";
          console.error("Task Failed:", errorMsg);
          setError(errorMsg);
          setTaskId(null);
        }
      } catch(e) {
        console.error(e);
        setError("Error fetching task status");
        setTaskId(null);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId, step]);

  const runTask = async (mode, params) => {
    setError("");
    const res = await fetch('/api/apps/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        app_name: appName,
        params: { app_mode: mode, ...params }
      })
    });
    const data = await res.json();
    setTaskId(data.task_id);
    if (addTask) {
      addTask({ id: data.task_id, name: `Adv Wan Animate - ${mode}`, status: 'queued', progress: 0 });
    }
  };

  const handleFileUpload = async (e, setPathCallback) => {
    const file = e.target.files[0];
    if (!file) return;

    // Web browsers do not expose absolute paths (file.path) for security reasons
    // unless running inside Electron. Therefore, we must upload it to get a server path.
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (data.status === 'success') {
            setPathCallback(data.file_path);
        } else {
            console.error("Upload failed", data);
        }
    } catch(err) {
        console.error("Upload error", err);
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
    
    setPoints([...points, [x, y]]);
    setLabels([...labels, isRightClick ? 0 : 1]);
  };

  const clearPoints = () => {
    setPoints([]);
    setLabels([]);
    setPreviewMask("");
  };

  return (
    <div style={{padding: '20px', fontFamily: 'sans-serif', background: '#0d0d0d', borderRadius: '8px', color: '#f0f0f0'}}>
      <h2>Advanced Wan Animate</h2>
      {taskId && <div style={{color: '#007bff', marginBottom: '10px'}}>Task running... ({taskStatus})</div>}
      {error && <div style={{color: '#ff4d4f', marginBottom: '10px', padding: '10px', background: 'rgba(255, 77, 79, 0.1)', border: '1px solid #ff4d4f', borderRadius: '4px'}}>Error: {error}</div>}
      
      <div style={{display: 'flex', gap: '10px', marginBottom: '20px', alignItems: 'center', justifyContent: 'space-between'}}>
        <div style={{display: 'flex', gap: '10px'}}>
            {["0: Init", "1: Segment", "2: Matte", "3: Pose", "4: Animate", "5: Output"].map((label, s) => (
            <button 
                key={s} 
                onClick={() => setStep(s)} 
                disabled={s > step && s !== 0} 
                style={{
                padding: '8px 12px', 
                cursor: (s > step && s !== 0) ? 'not-allowed' : 'pointer',
                background: step === s ? '#007bff' : '#222',
                color: step === s ? '#f0f0f0' : '#888',
                border: '1px solid #1a1a1a',
                borderRadius: '4px'
                }}
            >
                {label}
            </button>
            ))}
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
            <label style={{fontSize: '14px', color: '#f0f0f0'}}>Upload files to server</label>
            <input 
                type="checkbox" 
                checked={isRemote} 
                onChange={(e) => setIsRemote(e.target.checked)} 
                style={{cursor: 'pointer'}}
            />
        </div>
      </div>

      <div style={{background: '#121212', padding: '20px', borderRadius: '8px', border: '1px solid #1a1a1a'}}>
        {step === 0 && (
          <div>
            <h3>Step 0: Select Video</h3>
            <div style={{marginBottom: '10px'}}>
              <label style={{display: 'block', marginBottom: '5px'}}>{isRemote ? "Select Video to Upload:" : "Select Local Video:"}</label>
              <input 
                  type="file" 
                  accept="video/*"
                  onChange={e => handleFileUpload(e, setInputVideo)} 
                  style={{width: '100%', padding: '8px', background: '#050505', color: '#f0f0f0', border: '1px solid #1a1a1a', borderRadius: '4px'}} 
              />
              {inputVideo && <div style={{marginTop: '5px', fontSize: '12px', color: '#888'}}>Selected/Uploaded Path: {inputVideo}</div>}
              {isUploading && <div style={{marginTop: '5px', fontSize: '12px', color: '#007bff'}}>Uploading...</div>}
            </div>
            <button onClick={() => runTask("0_init", { input_video: inputVideo })} disabled={!!taskId || !inputVideo || isUploading} style={{padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
              Load Video
            </button>
          </div>
        )}

        {step === 1 && (
          <div>
            <h3>Step 1: Segment Character</h3>
            <p>Left click for foreground, right click for background.</p>
            <div style={{display: 'flex', gap: '20px'}}>
              <div>
                <h4>Original Frame</h4>
                {firstFrame && (
                  <div style={{position: 'relative', display: 'inline-block'}}>
                    <img 
                      ref={imgRef}
                      src={resolveMediaUrl(firstFrame)}
                      onClick={handleImageClick}
                      onContextMenu={handleImageClick}
                      style={{maxWidth: '400px', cursor: 'crosshair', border: '2px solid #1a1a1a', borderRadius: '4px'}}
                    />
                    {points.map((p, i) => (
                      <div key={i} style={{
                        position: 'absolute',
                        left: `${(p[0] / imgRef.current?.naturalWidth) * 100}%`,
                        top: `${(p[1] / imgRef.current?.naturalHeight) * 100}%`,
                        width: '8px', height: '8px', borderRadius: '50%',
                        background: labels[i] === 1 ? 'green' : 'red',
                        transform: 'translate(-50%, -50%)', pointerEvents: 'none'
                      }}></div>
                    ))}
                  </div>
                )}
                <div style={{marginTop: '10px', display: 'flex', gap: '10px'}}>
                  <button onClick={() => runTask("1_segment", { 
                    input_video: inputVideo, session_id: sessionId, points: JSON.stringify(points), labels: JSON.stringify(labels) 
                  })} disabled={!!taskId || points.length === 0} style={{padding: '8px 16px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>Generate Mask</button>
                  <button onClick={clearPoints} style={{padding: '8px 16px', background: '#222', color: '#f0f0f0', border: '1px solid #1a1a1a', borderRadius: '4px', cursor: 'pointer'}}>Clear Points</button>
                </div>
              </div>
              <div>
                <h4>Mask Preview</h4>
                {previewMask && <img src={resolveMediaUrl(previewMask)} style={{maxWidth: '400px', border: '1px solid #1a1a1a', borderRadius: '4px'}} />}
              </div>
            </div>
            {previewMask && (
              <button onClick={() => setStep(2)} disabled={!!taskId} style={{marginTop: '20px', padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
                Next Step (Matting)
              </button>
            )}
          </div>
        )}

        {step === 2 && (
          <div>
            <h3>Step 2: Generate Character and Mask Videos</h3>
            <p>This will use the mask from Step 1 to extract the character.</p>
            <button onClick={() => runTask("2_matte", { input_video: inputVideo, session_id: sessionId })} disabled={!!taskId} style={{padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
              Run Matting
            </button>
            <div style={{marginTop: '20px'}}>
              {fgVideo && <div><strong>Foreground Video:</strong> {fgVideo}</div>}
              {maskVideo && <div><strong>Mask Video:</strong> {maskVideo}</div>}
            </div>
            {fgVideo && (
              <button onClick={() => setStep(3)} disabled={!!taskId} style={{marginTop: '20px', padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
                Next Step (Pose)
              </button>
            )}
          </div>
        )}

        {step === 3 && (
          <div>
            <h3>Step 3: Generate Pose and Face Videos</h3>
            <button onClick={() => runTask("3_pose", { input_video: inputVideo })} disabled={!!taskId} style={{padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
              Run Pose Extraction
            </button>
            <div style={{marginTop: '20px'}}>
              {poseVideo && <div><strong>Pose Video:</strong> {poseVideo}</div>}
              {faceVideo && <div><strong>Face Video:</strong> {faceVideo}</div>}
            </div>
            {poseVideo && (
              <button onClick={() => setStep(4)} disabled={!!taskId} style={{marginTop: '20px', padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
                Next Step (Animate)
              </button>
            )}
          </div>
        )}

        {step === 4 && (
          <div>
            <h3>Step 4: Animate</h3>
            <div style={{display: 'flex', flexDirection: 'column', gap: '10px', maxWidth: '500px'}}>
              <div>
                <label style={{display: 'block', fontSize: '12px', color: '#888'}}>Positive Prompt:</label>
                <textarea value={positive} onChange={e => setPositive(e.target.value)} rows={3} style={{width: '100%', padding: '8px', background: '#050505', color: '#f0f0f0', border: '1px solid #1a1a1a', borderRadius: '4px'}} />
              </div>
              <div>
                <label style={{display: 'block', fontSize: '12px', color: '#888'}}>Negative Prompt:</label>
                <textarea value={negative} onChange={e => setNegative(e.target.value)} rows={3} style={{width: '100%', padding: '8px', background: '#050505', color: '#f0f0f0', border: '1px solid #1a1a1a', borderRadius: '4px'}} />
              </div>
              <div>
                <label style={{display: 'block', fontSize: '12px', color: '#888'}}>{isRemote ? "Upload Reference Image:" : "Select Local Reference Image:"}</label>
                <input 
                    type="file" 
                    accept="image/*"
                    onChange={e => handleFileUpload(e, setReferenceImage)} 
                    style={{width: '100%', padding: '8px', background: '#050505', color: '#f0f0f0', border: '1px solid #1a1a1a', borderRadius: '4px'}} 
                />
                {referenceImage && <div style={{marginTop: '5px', fontSize: '12px', color: '#888'}}>Selected/Uploaded Path: {referenceImage}</div>}
              </div>
              
              <button onClick={() => runTask("4_animate", { 
                bg_video: inputVideo, mask_video: maskVideo, pose_video: poseVideo, face_video: faceVideo,
                reference_image: referenceImage, positive: positive, negative: negative,
                seed: 42, steps: 20, cfg: 6.0
              })} disabled={!!taskId} style={{marginTop: '10px', padding: '10px 20px', background: '#007bff', color: '#f0f0f0', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
                Run Final Animation
              </button>
            </div>
          </div>
        )}

        {step === 5 && (
          <div>
            <h3>Final Output</h3>
            {finalVideo ? (
              <div>
                <p>Final Video generated successfully:</p>
                <video controls style={{maxWidth: '100%', border: '2px solid #1a1a1a', borderRadius: '8px'}} src={resolveMediaUrl(finalVideo)}></video>
              </div>
            ) : (
              <p>No final video yet.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AdvancedWanAnimateUI;
