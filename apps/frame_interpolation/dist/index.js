(function(React2) {
  "use strict";
  const ModelSelector = window.ModelSelector || (() => null);
  const FrameInterpolationUI = ({ appName = "Frame Interpolation", appDefinition }) => {
    const [taskId, setTaskId] = React2.useState(null);
    const [taskStatus, setTaskStatus] = React2.useState("");
    const [error, setError] = React2.useState("");
    const [models, setModels] = React2.useState({});
    const [keyframes, setKeyframes] = React2.useState(["", ""]);
    const [prompts, setPrompts] = React2.useState(["smooth transition"]);
    const [durations, setDurations] = React2.useState([5]);
    const [uploadingIndex, setUploadingIndex] = React2.useState(null);
    const [height, setHeight] = React2.useState(512);
    const [width, setWidth] = React2.useState(512);
    const [finalVideo, setFinalVideo] = React2.useState("");
    const keyframeInputRefs = React2.useRef([]);
    const actions = window.useTaskActions ? window.useTaskActions() : null;
    const addTask = actions ? actions.addTask : null;
    const { addToast } = window.useToast ? window.useToast() : { addToast: (msg) => alert(msg.message || msg) };
    const resolveMediaUrl = (path) => {
      if (!path) return "";
      if (path.startsWith("http")) return path;
      if (path.startsWith("/api/outputs")) return path;
      return `/api/outputs/${path}`;
    };
    const addKeyframe = () => {
      setKeyframes([...keyframes, ""]);
      setPrompts([...prompts, "smooth transition"]);
      setDurations([...durations, 5]);
    };
    const removeKeyframe = (index) => {
      if (keyframes.length <= 2) return;
      const newKeyframes = [...keyframes];
      newKeyframes.splice(index, 1);
      setKeyframes(newKeyframes);
      const newPrompts = [...prompts];
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
      if (keyframes.some((kf) => !kf)) {
        addToast({ message: "Please upload all keyframes", type: "error" });
        return;
      }
      setError("");
      try {
        const res = await fetch("/api/apps/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            app_name: appName,
            params: {
              ...models,
              keyframes,
              prompts,
              durations,
              seed: -1,
              steps: 20,
              cfg: 6,
              width,
              height
            }
          })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setTaskId(data.task_id);
        setFinalVideo("");
        if (addTask) {
          addTask({
            id: data.task_id,
            name: appName,
            status: "queued",
            progress: 0
          });
        }
      } catch (err) {
        setError(err.message || "Failed to start task");
      }
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
            setFinalVideo(output);
          } else if (data.status === "failed") {
            const errorMsg = ((_a = data.data) == null ? void 0 : _a.err_message) || "Task failed";
            setError(errorMsg);
            setTaskId(null);
          }
        } catch (e) {
          console.error(e);
          setError("Error fetching task status");
          setTaskId(null);
        }
      }, 1e3);
      return () => clearInterval(interval);
    }, [taskId]);
    const handleFileUpload = async (file, index) => {
      if (!file) return;
      setUploadingIndex(index);
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (data.status === "success") {
          updateKeyframe(index, data.file_path);
        }
      } catch (err) {
        console.error("Upload error", err);
      } finally {
        setUploadingIndex(null);
      }
    };
    const statusClass = taskStatus === "failed" ? "failed" : taskId ? "running" : finalVideo ? "done" : "idle";
    return /* @__PURE__ */ React2.createElement("div", { className: "awa-page" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-bg" }), /* @__PURE__ */ React2.createElement("div", { className: "awa-shell" }, /* @__PURE__ */ React2.createElement("header", { className: "awa-header" }, /* @__PURE__ */ React2.createElement("div", null, /* @__PURE__ */ React2.createElement("h1", null, "Frame Interpolation"), /* @__PURE__ */ React2.createElement("p", { style: { margin: "4px 0 0 0", fontSize: "14px", color: "var(--awa-muted)" } }, "Create complex video transitions by interpolating through multiple keyframes")), /* @__PURE__ */ React2.createElement("div", { className: `awa-run-status ${statusClass}` }, /* @__PURE__ */ React2.createElement("span", null, taskId ? "Running" : "Status"), /* @__PURE__ */ React2.createElement("strong", null, taskId ? taskStatus || "queued" : finalVideo ? "ready" : "idle"))), error && /* @__PURE__ */ React2.createElement("div", { className: "awa-alert", style: { animationDelay: "120ms" } }, error), /* @__PURE__ */ React2.createElement("main", { className: "awa-grid" }, /* @__PURE__ */ React2.createElement("section", { className: "awa-card full-width" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-card-head" }, /* @__PURE__ */ React2.createElement("h2", null, "1. Keyframe Sequence")), /* @__PURE__ */ React2.createElement("div", { className: "keyframes-scroll-container" }, /* @__PURE__ */ React2.createElement("div", { className: "keyframes-sequence" }, keyframes.map((kf, index) => /* @__PURE__ */ React2.createElement(React2.Fragment, { key: index }, /* @__PURE__ */ React2.createElement("div", { className: "keyframe-block" }, /* @__PURE__ */ React2.createElement("div", { className: "keyframe-label" }, /* @__PURE__ */ React2.createElement("span", null, "Frame ", index + 1), keyframes.length > 2 && /* @__PURE__ */ React2.createElement("button", { className: "remove-kf-btn", onClick: () => removeKeyframe(index) }, "×")), /* @__PURE__ */ React2.createElement(
      "div",
      {
        className: "awa-placeholder kf-preview",
        onClick: () => keyframeInputRefs.current[index].click()
      },
      uploadingIndex === index ? /* @__PURE__ */ React2.createElement("div", { className: "awa-spinner" }) : kf ? /* @__PURE__ */ React2.createElement("img", { src: resolveMediaUrl(kf), alt: `Frame ${index + 1}` }) : /* @__PURE__ */ React2.createElement("span", null, "Click to upload")
    ), /* @__PURE__ */ React2.createElement(
      "input",
      {
        ref: (el) => keyframeInputRefs.current[index] = el,
        type: "file",
        accept: "image/*",
        style: { display: "none" },
        onChange: (e) => {
          handleFileUpload(e.target.files[0], index);
          e.target.value = null;
        }
      }
    )), index < keyframes.length - 1 && /* @__PURE__ */ React2.createElement("div", { className: "segment-settings" }, /* @__PURE__ */ React2.createElement("div", { className: "segment-arrow" }, "→"), /* @__PURE__ */ React2.createElement("div", { className: "segment-inputs" }, /* @__PURE__ */ React2.createElement(
      "textarea",
      {
        className: "awa-textarea segment-prompt",
        placeholder: "Prompt for this transition...",
        value: prompts[index],
        onChange: (e) => updatePrompt(index, e.target.value),
        rows: 2
      }
    ), /* @__PURE__ */ React2.createElement("div", { className: "duration-selector" }, /* @__PURE__ */ React2.createElement("label", null, "Duration:"), /* @__PURE__ */ React2.createElement(
      "select",
      {
        value: durations[index],
        onChange: (e) => updateDuration(index, e.target.value),
        className: "awa-input",
        style: { marginBottom: 0, padding: "4px 8px", width: "auto" }
      },
      /* @__PURE__ */ React2.createElement("option", { value: 3 }, "3s"),
      /* @__PURE__ */ React2.createElement("option", { value: 4 }, "4s"),
      /* @__PURE__ */ React2.createElement("option", { value: 5 }, "5s")
    )))))), /* @__PURE__ */ React2.createElement("button", { className: "add-keyframe-card", onClick: addKeyframe }, /* @__PURE__ */ React2.createElement("div", { className: "plus-icon" }, "+"), /* @__PURE__ */ React2.createElement("span", null, "Add Frame"))))), /* @__PURE__ */ React2.createElement("section", { className: "awa-card full-width" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-card-head" }, /* @__PURE__ */ React2.createElement("h2", null, "2. Global Settings & Output")), /* @__PURE__ */ React2.createElement("div", { className: "awa-init-layout" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-init-sidebar" }, ModelSelector && /* @__PURE__ */ React2.createElement(ModelSelector, { appDefinition, onChange: setModels }), /* @__PURE__ */ React2.createElement("div", { className: "awa-row" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-col" }, /* @__PURE__ */ React2.createElement("label", { className: "awa-label" }, "Width"), /* @__PURE__ */ React2.createElement(
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
    ))), /* @__PURE__ */ React2.createElement(
      "button",
      {
        className: "awa-btn primary full-width",
        type: "button",
        onClick: runTask,
        disabled: !!taskId || uploadingIndex !== null,
        style: { marginTop: "10px", width: "100%" }
      },
      "Generate Video Sequence"
    ), /* @__PURE__ */ React2.createElement("p", { style: { marginTop: "10px", fontSize: "12px", color: "var(--awa-muted)", textAlign: "center" } }, "Note: This uses Wan VACE 14B, which can take up to 6 minutes to generate a 5-second video on an RTX A5000")), /* @__PURE__ */ React2.createElement("div", { className: "awa-seg-main" }, /* @__PURE__ */ React2.createElement("div", { className: "awa-media-panel" }, /* @__PURE__ */ React2.createElement("h3", null, "Final Interpolated Video"), finalVideo ? /* @__PURE__ */ React2.createElement("video", { className: "awa-final-video", controls: true, src: resolveMediaUrl(finalVideo), autoPlay: true, loop: true }) : /* @__PURE__ */ React2.createElement("div", { className: "awa-placeholder", style: { height: "300px" } }, "Final sequence will appear here after generation"))))))));
  };
  window.ExivPlugins = window.ExivPlugins || {};
  window.ExivPlugins["Frame Interpolation"] = FrameInterpolationUI;
  console.log("Frame Interpolation Plugin Registered!");
})(React);
