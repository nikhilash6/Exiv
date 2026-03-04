(function(React2) {
  "use strict";
  const ModelSelector = window.ModelSelector || (() => null);
  const AdvancedWanAnimateUI = ({ appName = "Character Replace", appDefinition }) => {
    const [step, setStep] = React2.useState(0);
    const [taskId, setTaskId] = React2.useState(null);
    const [taskStatus, setTaskStatus] = React2.useState("");
    const [error, setError] = React2.useState("");
    const [isAutoChaining, setIsAutoChaining] = React2.useState(false);
    const [models, setModels] = React2.useState({});
    const [inputVideo, setInputVideo] = React2.useState("");
    const [sessionId, setSessionId] = React2.useState("");
    const [firstFrame, setFirstFrame] = React2.useState("");
    const [isUploading, setIsUploading] = React2.useState(false);
    const [points, setPoints] = React2.useState([]);
    const [labels, setLabels] = React2.useState([]);
    const [previewMask, setPreviewMask] = React2.useState("");
    const [fgVideo, setFgVideo] = React2.useState("");
    const [maskVideo, setMaskVideo] = React2.useState("");
    const [poseVideo, setPoseVideo] = React2.useState("");
    const [faceVideo, setFaceVideo] = React2.useState("");
    const [finalVideo, setFinalVideo] = React2.useState("");
    const [positive, setPositive] = React2.useState("a girl talking");
    const [negative, setNegative] = React2.useState("bad quality");
    const [referenceImage, setReferenceImage] = React2.useState("");
    const [height, setHeight] = React2.useState(640);
    const [width, setWidth] = React2.useState(640);
    const [videoDuration, setVideoDuration] = React2.useState(0);
    const [videoLength, setVideoLength] = React2.useState(5);
    const imgRef = React2.useRef(null);
    const videoInputRef = React2.useRef(null);
    const refImageInputRef = React2.useRef(null);
    const actions = window.useTaskActions ? window.useTaskActions() : null;
    const addTask = actions ? actions.addTask : null;
    const { addToast } = window.useToast ? window.useToast() : { addToast: () => alert("Please create mask and upload the reference image") };
    const resolveMediaUrl = (path) => {
      if (!path) return "";
      if (path.startsWith("http")) return path;
      if (path.startsWith("/api/outputs")) return path;
      return `/api/outputs/${path}`;
    };
    const runTask = async (mode, params) => {
      setError("");
      const res = await fetch("/api/apps/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_name: appName,
          params: { app_mode: mode, ...models, ...params }
        })
      });
      const data = await res.json();
      setTaskId(data.task_id);
      if (addTask) {
        let taskName = `Character Replace - ${mode}`;
        if (mode === "0_init") taskName = "Extract frame";
        else if (mode === "1_segment") taskName = "Segment mask";
        else if (mode === "4_animate") taskName = "Animate";
        addTask({
          id: data.task_id,
          name: taskName,
          status: "queued",
          progress: 0
        });
      }
      return data.task_id;
    };
    React2.useEffect(() => {
      if (!taskId) return;
      const interval = setInterval(async () => {
        var _a;
        try {
          const res = await fetch(`/status/${taskId}`);
          const data = await res.json();
          setTaskStatus(data.status);
          if (data.status === "completed") {
            const output = data.output["1"];
            setTaskId(null);
            setError("");
            if (step === 0) {
              setFirstFrame(output.first_frame);
              setSessionId(output.session_id);
              setStep(1);
            } else if (step === 1) {
              setPreviewMask(output.preview);
            } else if (step === 4) {
              setFinalVideo(output.final_video);
              if (output.fg_video) setFgVideo(output.fg_video);
              if (output.mask_video) setMaskVideo(output.mask_video);
              if (output.pose_video) setPoseVideo(output.pose_video);
              if (output.face_video) setFaceVideo(output.face_video);
              setStep(5);
              setIsAutoChaining(false);
            }
          } else if (data.status === "failed") {
            const errorMsg = ((_a = data.data) == null ? void 0 : _a.err_message) || "Task failed";
            setError(errorMsg);
            setTaskId(null);
            setIsAutoChaining(false);
          }
        } catch (e) {
          console.error(e);
          setError("Error fetching task status");
          setTaskId(null);
          setIsAutoChaining(false);
        }
      }, 1e3);
      return () => clearInterval(interval);
    }, [taskId, step, isAutoChaining, inputVideo, sessionId, positive, referenceImage, width, height, videoLength, fgVideo, maskVideo]);
    const handleRenderFinal = () => {
      if (!inputVideo || !previewMask) {
        if (window.useToast) {
          addToast({ message: "please upload a video and create a mask", type: "error" });
        } else {
          alert("please upload a video and create a mask");
        }
        return;
      }
      if (!referenceImage) {
        if (window.useToast) {
          addToast({ message: "Please upload the reference image", type: "error" });
        } else {
          alert("Please upload the reference image");
        }
        return;
      }
      setStep(4);
      if (window.useToast) {
        addToast({ message: "Animate task queued", type: "success" });
      }
      runTask("4_animate", {
        input_video: inputVideo,
        session_id: sessionId,
        bg_video: "",
        mask_video: "",
        pose_video: "",
        face_video: "",
        reference_image: referenceImage,
        positive,
        negative: "",
        seed: -1,
        steps: 4,
        cfg: 1,
        width,
        height,
        frame_count: Math.ceil(videoLength * 16)
      });
    };
    const handleFileUpload = async (file, setPathCallback, autoInit = false) => {
      if (!file) return;
      if (autoInit) {
        clearPoints();
        setFirstFrame("");
        setFinalVideo("");
        setFgVideo("");
        setPoseVideo("");
        setMaskVideo("");
        setFaceVideo("");
        setTaskId(null);
        setTaskStatus("");
      }
      if (file.type.startsWith("video/")) {
        const video = document.createElement("video");
        video.preload = "metadata";
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
      formData.append("file", file);
      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (data.status === "success") {
          setPathCallback(data.file_path);
          if (autoInit) {
            setStep(0);
            runTask("0_init", { input_video: data.file_path });
          }
        }
      } catch (err) {
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
      const isRightClick = e.type === "contextmenu";
      if (isRightClick) e.preventDefault();
      setPoints((prev) => [...prev, [x, y]]);
      setLabels((prev) => [...prev, isRightClick ? 0 : 1]);
    };
    const clearPoints = () => {
      setPoints([]);
      setLabels([]);
      setPreviewMask("");
    };
    const statusClass = taskStatus === "failed" ? "failed" : taskId && taskStatus !== "completed" ? "running" : finalVideo ? "done" : "idle";
    const getStatusText = () => {
      if (taskStatus === "failed") return "failed";
      if (taskId && taskStatus !== "completed") return taskStatus || "running";
      if (finalVideo) return "ready";
      return "idle";
    };
    return /* @__PURE__ */ React2.createElement("div", { className: "awa-page" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-bg" }), /* @__PURE__ */ React2.createElement("div", { className: "awa-shell" }, /* @__PURE__ */ React2.createElement("header", { className: "awa-header" }, /* @__PURE__ */ React2.createElement("div", null, /* @__PURE__ */ React2.createElement("h1", null, "Character Replace"), /* @__PURE__ */ React2.createElement("p", { style: { margin: "4px 0 0 0", fontSize: "14px", color: "var(--awa-muted)" } }, "Replaces a character from the input video to the one provided by you")), /* @__PURE__ */ React2.createElement("div", { className: `awa-run-status ${statusClass}` }, /* @__PURE__ */ React2.createElement("span", null, "Status"), /* @__PURE__ */ React2.createElement("strong", null, getStatusText()))), error && /* @__PURE__ */ React2.createElement("div", { className: "awa-alert", style: { animationDelay: "120ms" } }, error), /* @__PURE__ */ React2.createElement("main", { className: "awa-grid" }, /* @__PURE__ */ React2.createElement("section", { className: "awa-card full-width" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-card-head" }, /* @__PURE__ */ React2.createElement("h2", null, "1. Initialize & Segment"), /* @__PURE__ */ React2.createElement("span", { className: previewMask ? "pill done" : "pill" }, previewMask ? "Mask Ready" : firstFrame ? "Awaiting Mask" : "Ready")), /* @__PURE__ */ React2.createElement("div", { className: "awa-init-layout", style: { gap: "24px" } }, /* @__PURE__ */ React2.createElement("div", { className: "awa-seg-main", style: { flex: "1 1 50%", maxWidth: "50%", paddingRight: "12px", borderRight: "1px solid var(--awa-card-border)" } }, /* @__PURE__ */ React2.createElement("h3", { style: { fontSize: "14px", marginBottom: "8px" } }, "Load Source Video"), /* @__PURE__ */ React2.createElement("p", { style: { fontSize: "12px", color: "var(--awa-muted)", marginBottom: "4px" } }, "Pick source video and initialize session."), /* @__PURE__ */ React2.createElement("p", { style: { fontSize: "12px", color: "var(--awa-muted)", marginBottom: "16px" } }, "* Videos of length 5-10 secs are preferred."), /* @__PURE__ */ React2.createElement(
      "input",
      {
        ref: videoInputRef,
        type: "file",
        accept: "video/*",
        style: { display: "none" },
        onChange: (e) => {
          handleFileUpload(e.target.files[0], setInputVideo, true);
          e.target.value = null;
        }
      }
    ), /* @__PURE__ */ React2.createElement(
      "button",
      {
        className: "awa-btn primary",
        type: "button",
        onClick: () => videoInputRef.current.click(),
        disabled: !!taskId || isUploading
      },
      "Load Video"
    ), /* @__PURE__ */ React2.createElement("p", { className: "awa-path", style: { marginTop: "12px", marginBottom: "40px" } }, inputVideo ? `Loaded: ${inputVideo}` : "No video selected"), /* @__PURE__ */ React2.createElement("h3", { style: { fontSize: "14px", marginBottom: "8px" } }, "Mask Generation"), /* @__PURE__ */ React2.createElement("p", { style: { fontSize: "12px", color: "var(--awa-muted)", marginBottom: "16px" } }, "Click to mark a point, mask will be generated for this marked segment"), /* @__PURE__ */ React2.createElement("div", { className: "awa-media-row", style: { gridTemplateColumns: "1fr", margin: "0 0 20px 0" } }, /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, previewMask ? "Mask Preview (Click to refine)" : "Frame"), isUploading || taskId && (step === 0 || step === 1) ? /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder", style: { height: "300px" } }, /* @__PURE__ */ React2.createElement("div", { className: "awa-spinner" })) : firstFrame ? /* @__PURE__ */ React2.createElement("div", { className: "awa-image-wrap" }, /* @__PURE__ */ React2.createElement(
      "img",
      {
        ref: imgRef,
        src: resolveMediaUrl(previewMask || firstFrame),
        onClick: handleImageClick,
        onContextMenu: handleImageClick,
        alt: "preview"
      }
    ), points.map((p, i) => {
      var _a, _b;
      return /* @__PURE__ */ React2.createElement(
        "span",
        {
          key: `${p[0]}-${p[1]}-${i}`,
          className: `awa-point ${labels[i] === 1 ? "fg" : "bg"}`,
          style: {
            left: `${p[0] / (((_a = imgRef.current) == null ? void 0 : _a.naturalWidth) || 1) * 100}%`,
            top: `${p[1] / (((_b = imgRef.current) == null ? void 0 : _b.naturalHeight) || 1) * 100}%`
          }
        }
      );
    })) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder", style: { height: "300px" } }, "Initialize video to see frame"))), /* @__PURE__ */ React2.createElement("div", { className: "awa-actions" }, /* @__PURE__ */ React2.createElement(
      "button",
      {
        className: "awa-btn primary",
        type: "button",
        onClick: () => runTask("1_segment", {
          input_video: inputVideo,
          session_id: sessionId,
          points: JSON.stringify(points),
          labels: JSON.stringify(labels)
        }),
        disabled: !!taskId || points.length === 0 || !sessionId
      },
      taskId && step === 1 ? "Generating..." : "Generate Mask"
    ), /* @__PURE__ */ React2.createElement("button", { className: "awa-btn ghost", type: "button", onClick: clearPoints }, "Clear Mask"))), /* @__PURE__ */ React2.createElement("div", { className: "awa-init-sidebar", style: { flex: "1 1 50%", maxWidth: "50%", paddingLeft: "12px" } }, /* @__PURE__ */ React2.createElement("div", { style: { marginBottom: "40px" } }, /* @__PURE__ */ React2.createElement("h3", { style: { fontSize: "14px", marginBottom: "8px" } }, "Select the character to replace"), /* @__PURE__ */ React2.createElement("p", { style: { fontSize: "12px", color: "var(--awa-muted)", marginBottom: "16px" } }, "it should have a white background"), /* @__PURE__ */ React2.createElement(
      "input",
      {
        ref: refImageInputRef,
        type: "file",
        accept: "image/*",
        style: { display: "none" },
        onChange: (e) => {
          handleFileUpload(e.target.files[0], setReferenceImage);
          e.target.value = null;
        }
      }
    ), /* @__PURE__ */ React2.createElement(
      "div",
      {
        className: "awa-placeholder",
        style: { width: "100%", height: "300px", cursor: "pointer", overflow: "hidden" },
        onClick: () => refImageInputRef.current.click()
      },
      referenceImage ? /* @__PURE__ */ React2.createElement("img", { src: resolveMediaUrl(referenceImage), alt: "reference", style: { width: "100%", height: "100%", objectFit: "contain" } }) : /* @__PURE__ */ React2.createElement("span", null, "Click to select reference image")
    ))))), /* @__PURE__ */ React2.createElement("section", { className: "awa-card full-width", style: { animationDelay: "260ms" } }, /* @__PURE__ */ React2.createElement("div", { className: "awa-card-head" }, /* @__PURE__ */ React2.createElement("h2", null, "2. Animate"), /* @__PURE__ */ React2.createElement("span", { className: finalVideo ? "pill done" : "pill" }, finalVideo ? "Rendered" : taskId && isAutoChaining ? "Generating..." : "Ready to Run")), /* @__PURE__ */ React2.createElement("div", { className: "awa-init-layout" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-init-sidebar" }, /* @__PURE__ */ React2.createElement("p", null, "Configure prompts and reference image, then render output."), ModelSelector && /* @__PURE__ */ React2.createElement(ModelSelector, { appDefinition, onChange: setModels }), /* @__PURE__ */ React2.createElement("label", { className: "awa-label", htmlFor: "positivePrompt" }, "Prompt"), /* @__PURE__ */ React2.createElement(
      "textarea",
      {
        id: "positivePrompt",
        className: "awa-textarea",
        value: positive,
        onChange: (e) => setPositive(e.target.value),
        rows: 3
      }
    ), /* @__PURE__ */ React2.createElement("div", { className: "awa-row" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-col" }, /* @__PURE__ */ React2.createElement("label", { className: "awa-label" }, "Width"), /* @__PURE__ */ React2.createElement(
      "input",
      {
        type: "number",
        className: "awa-input",
        value: width,
        onChange: (e) => setWidth(parseInt(e.target.value))
      }
    )), /* @__PURE__ */ React2.createElement("div", { className: "awa-col" }, /* @__PURE__ */ React2.createElement("label", { className: "awa-label" }, "Height"), /* @__PURE__ */ React2.createElement(
      "input",
      {
        type: "number",
        className: "awa-input",
        value: height,
        onChange: (e) => setHeight(parseInt(e.target.value))
      }
    ))), /* @__PURE__ */ React2.createElement("div", { style: { marginBottom: "14px" } }, /* @__PURE__ */ React2.createElement("label", { className: "awa-label" }, "Video Length: ", videoLength, " secs"), /* @__PURE__ */ React2.createElement(
      "input",
      {
        type: "range",
        className: "awa-input",
        style: { padding: 0 },
        min: videoDuration > 0 ? Math.min(videoDuration, 5) : 5,
        max: videoDuration > 0 ? Math.max(videoDuration, 10) : 10,
        step: 1,
        value: videoLength,
        onChange: (e) => setVideoLength(parseFloat(e.target.value))
      }
    )), /* @__PURE__ */ React2.createElement(
      "button",
      {
        className: "awa-btn primary",
        type: "button",
        onClick: handleRenderFinal,
        disabled: !!taskId || isUploading
      },
      "Render Final Animation"
    )), /* @__PURE__ */ React2.createElement("div", { className: "awa-seg-main" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, "Final Output"), finalVideo ? /* @__PURE__ */ React2.createElement("video", { className: "awa-final-video", controls: true, src: resolveMediaUrl(finalVideo) }) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder" }, "Final rendered video will appear here")), (fgVideo || poseVideo || maskVideo || faceVideo) && /* @__PURE__ */ React2.createElement("div", { className: "awa-intermediate-results", style: { marginTop: "32px", borderTop: "1px solid var(--awa-card-border)", paddingTop: "24px" } }, /* @__PURE__ */ React2.createElement("h2", { style: { fontSize: "14px", marginBottom: "16px", color: "var(--awa-muted)" } }, "Intermediate Generation Steps"), /* @__PURE__ */ React2.createElement("div", { className: "awa-media-row" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, "Foreground Matting"), fgVideo ? /* @__PURE__ */ React2.createElement("video", { controls: true, src: resolveMediaUrl(fgVideo) }) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder" }, "Processing...")), /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, "Background Mask"), maskVideo ? /* @__PURE__ */ React2.createElement("video", { controls: true, src: resolveMediaUrl(maskVideo) }) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder" }, "Processing..."))), /* @__PURE__ */ React2.createElement("div", { className: "awa-media-row", style: { marginTop: "16px" } }, /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, "Pose Estimation"), poseVideo ? /* @__PURE__ */ React2.createElement("video", { controls: true, src: resolveMediaUrl(poseVideo) }) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder" }, "Processing...")), /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, "Face Tracking"), faceVideo ? /* @__PURE__ */ React2.createElement("video", { controls: true, src: resolveMediaUrl(faceVideo) }) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder" }, "Processing..."))))))))));
  };
  window.ExivPlugins = window.ExivPlugins || {};
  window.ExivPlugins["Character Replace"] = AdvancedWanAnimateUI;
  console.log("Character Replace Plugin Registered!");
})(React);
