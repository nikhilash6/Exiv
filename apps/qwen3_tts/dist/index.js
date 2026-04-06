(function(React2) {
  "use strict";
  const DEFAULT_VOICES = {
    "calm_male.wav": { text: "Hello! I'm an AI assistant...", tags: "Male, 30s, calm and soothing voice", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/calm_male.wav" },
    "corporate_female.wav": { text: "Hello! I'm an AI assistant...", tags: "Female, 40s, corporate executive voice", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/corporate_female.wav" },
    "dramatic_male.wav": { text: "Hello! I'm an AI assistant...", tags: "Male, deep dramatic voice", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/dramatic_male.wav" },
    "enthusiastic_male.wav": { text: "Hello! I'm an AI assistant...", tags: "Young male, 20s, energetic", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/enthusiastic_male.wav" },
    "wise_female.wav": { text: "Hello! I'm an AI assistant...", tags: "Elderly female, wise ethereal", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/wise_female.wav" },
    "young_female.wav": { text: "Hello! I'm Vivian...", tags: "Young female, teens, energetic", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/young_female.wav" },
    "young_male.wav": { text: "Hello! I'm Aiden...", tags: "Young male, teens, casual gamer", language: "English", url: "https://huggingface.co/buckets/lonesamurai/sample_voices/resolve/young_male.wav" }
  };
  const LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Arabic", "Hindi"];
  const SearchIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("circle", { cx: "11", cy: "11", r: "8" }), /* @__PURE__ */ React2.createElement("path", { d: "m21 21-4.3-4.3" }));
  const MicIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("path", { d: "M12 19v3" }), /* @__PURE__ */ React2.createElement("path", { d: "M19 10v2a7 7 0 0 1-14 0v-2" }), /* @__PURE__ */ React2.createElement("rect", { x: "9", y: "2", width: "6", height: "13", rx: "3" }));
  const WaveformIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "24", height: "24", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("path", { d: "M2 10v4" }), /* @__PURE__ */ React2.createElement("path", { d: "M6 6v12" }), /* @__PURE__ */ React2.createElement("path", { d: "M10 3v18" }), /* @__PURE__ */ React2.createElement("path", { d: "M14 8v8" }), /* @__PURE__ */ React2.createElement("path", { d: "M18 5v14" }), /* @__PURE__ */ React2.createElement("path", { d: "M22 10v4" }));
  const PlusIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("line", { x1: "12", y1: "5", x2: "12", y2: "19" }), /* @__PURE__ */ React2.createElement("line", { x1: "5", y1: "12", x2: "19", y2: "12" }));
  const CloseIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("line", { x1: "18", y1: "6", x2: "6", y2: "18" }), /* @__PURE__ */ React2.createElement("line", { x1: "6", y1: "6", x2: "18", y2: "18" }));
  const PlayIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("polygon", { points: "5 3 19 12 5 21 5 3" }));
  const PauseIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("rect", { x: "6", y: "4", width: "4", height: "16" }), /* @__PURE__ */ React2.createElement("rect", { x: "14", y: "4", width: "4", height: "16" }));
  const SparklesIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("path", { d: "m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" }), /* @__PURE__ */ React2.createElement("path", { d: "M5 3v4" }), /* @__PURE__ */ React2.createElement("path", { d: "M19 17v4" }));
  const CheckIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "3" }, /* @__PURE__ */ React2.createElement("polyline", { points: "20 6 9 17 4 12" }));
  const VolumeIcon = () => /* @__PURE__ */ React2.createElement("svg", { width: "14", height: "14", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "2" }, /* @__PURE__ */ React2.createElement("polygon", { points: "11 5 6 9 2 9 2 15 6 15 11 19 11 5" }), /* @__PURE__ */ React2.createElement("path", { d: "M15.54 8.46a5 5 0 0 1 0 7.07" }));
  const UploadModal = ({ isOpen, onClose, onUpload }) => {
    const [name, setName] = React2.useState("");
    const [description, setDescription] = React2.useState("");
    const [file, setFile] = React2.useState(null);
    const [referenceText, setReferenceText] = React2.useState("");
    const [isUploading, setIsUploading] = React2.useState(false);
    if (!isOpen) return null;
    const handleSubmit = async (e) => {
      e.preventDefault();
      if (!name.trim() || !file) return;
      setIsUploading(true);
      const success = await onUpload({ name: name.trim(), description: description.trim() || name.trim(), referenceText: referenceText.trim(), file });
      setIsUploading(false);
      if (success) {
        setName("");
        setDescription("");
        setReferenceText("");
        setFile(null);
        onClose();
      }
    };
    return /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-modal-overlay", onClick: onClose }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-modal", onClick: (e) => e.stopPropagation() }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-modal-header" }, /* @__PURE__ */ React2.createElement("h3", null, "Add New Voice"), /* @__PURE__ */ React2.createElement("button", { onClick: onClose, className: "qwen3-tts-modal-close" }, /* @__PURE__ */ React2.createElement(CloseIcon, null))), /* @__PURE__ */ React2.createElement("form", { onSubmit: handleSubmit, className: "qwen3-tts-modal-form" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-form-group" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-form-label" }, "Voice Name *"), /* @__PURE__ */ React2.createElement("input", { type: "text", value: name, onChange: (e) => setName(e.target.value), placeholder: "e.g., my_custom_voice", className: "qwen3-tts-form-input", required: true })), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-form-group" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-form-label" }, "Description *"), /* @__PURE__ */ React2.createElement("input", { type: "text", value: description, onChange: (e) => setDescription(e.target.value), placeholder: "e.g., Deep male voice, professional tone", className: "qwen3-tts-form-input", required: true })), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-form-group" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-form-label" }, "Reference Text (optional - x-vector mode if empty)"), /* @__PURE__ */ React2.createElement("textarea", { value: referenceText, onChange: (e) => setReferenceText(e.target.value), placeholder: "The spoken text in the audio file...", className: "qwen3-tts-form-textarea", rows: 3 })), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-form-group" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-form-label" }, "Audio File (.wav) *"), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-file-input-wrapper" }, /* @__PURE__ */ React2.createElement("input", { type: "file", accept: ".wav,audio/wav", onChange: (e) => setFile(e.target.files[0]), className: "qwen3-tts-file-input", id: "voice-file", required: true }), /* @__PURE__ */ React2.createElement("label", { htmlFor: "voice-file", className: "qwen3-tts-file-label" }, file ? file.name : "Choose file..."))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-modal-actions" }, /* @__PURE__ */ React2.createElement("button", { type: "button", onClick: onClose, className: "qwen3-tts-btn qwen3-tts-btn-secondary" }, "Cancel"), /* @__PURE__ */ React2.createElement("button", { type: "submit", className: "qwen3-tts-btn qwen3-tts-btn-primary", disabled: isUploading || !name.trim() || !file }, isUploading ? /* @__PURE__ */ React2.createElement(React2.Fragment, null, /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-spinner-small" }), "Uploading...") : /* @__PURE__ */ React2.createElement(React2.Fragment, null, /* @__PURE__ */ React2.createElement(PlusIcon, null), "Add Voice"))))));
  };
  const VoiceCard = ({ voiceId, voiceData, isSelected, isPlaying, isLoading, onSelect, onPlay }) => {
    const formatVoiceName = (id) => id.replace(".wav", "").replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
    return /* @__PURE__ */ React2.createElement("button", { onClick: onSelect, className: `qwen3-tts-voice-card ${isSelected ? "qwen3-tts-voice-card-selected" : ""}` }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-voice-card-header" }, /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-voice-name" }, formatVoiceName(voiceId)), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-voice-actions" }, /* @__PURE__ */ React2.createElement("button", { onClick: (e) => {
      e.stopPropagation();
      onPlay(voiceId);
    }, className: `qwen3-tts-play-btn ${isPlaying ? "qwen3-tts-play-btn-playing" : ""} ${isLoading ? "qwen3-tts-play-btn-loading" : ""}` }, isLoading ? /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-spinner-tiny" }) : isPlaying ? /* @__PURE__ */ React2.createElement(PauseIcon, null) : /* @__PURE__ */ React2.createElement(PlayIcon, null)), isSelected && /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-selected-indicator" }, /* @__PURE__ */ React2.createElement(CheckIcon, null)))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-voice-tags" }, voiceData.tags), !voiceData.text && /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-xvector-badge" }, "X-Vector Mode"));
  };
  const OutputItem = ({ output }) => {
    const getAudioUrl = (path) => {
      if (!path) return "";
      if (path.startsWith("http")) return path;
      const filename = path.split("/").pop();
      return `/api/outputs/qwen3-tts/${filename}`;
    };
    return /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-output-item" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-output-item-header" }, /* @__PURE__ */ React2.createElement(VolumeIcon, null), /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-output-item-name" }, output.name)), /* @__PURE__ */ React2.createElement("audio", { src: getAudioUrl(output.path), controls: true, className: "qwen3-tts-output-audio", preload: "metadata" }));
  };
  const App = ({ appName = "Qwen3 TTS" }) => {
    const [voices, setVoices] = React2.useState(DEFAULT_VOICES);
    const [selectedVoice, setSelectedVoice] = React2.useState("calm_male.wav");
    const [text, setText] = React2.useState("");
    const [language, setLanguage] = React2.useState("English");
    const [taskId, setTaskId] = React2.useState(null);
    const [isGenerating, setIsGenerating] = React2.useState(false);
    const [searchQuery, setSearchQuery] = React2.useState("");
    const [isUploadModalOpen, setIsUploadModalOpen] = React2.useState(false);
    const [outputs, setOutputs] = React2.useState([]);
    const [playingVoice, setPlayingVoice] = React2.useState(null);
    const [downloadingVoice, setDownloadingVoice] = React2.useState(null);
    const voiceAudioRef = React2.useRef(new Audio());
    const actions = window.useTaskActions ? window.useTaskActions() : null;
    const addTask = actions ? actions.addTask : null;
    const updateTask = actions ? actions.updateTask : null;
    const { addToast } = window.useToast ? window.useToast() : { addToast: (msg) => console.log(msg) };
    const loadOutputs = React2.useCallback(async () => {
      try {
        const res = await fetch("/api/outputs?subfolder=qwen3-tts");
        if (!res.ok) throw new Error("Failed to fetch outputs");
        const data = await res.json();
        console.log("Outputs API response:", data);
        let files = Array.isArray(data) ? data : [];
        const qwen3Outputs = files.filter((f) => f.filename && f.filename.endsWith(".wav")).map((f, idx) => ({
          id: f.filename,
          path: `qwen3-tts/${f.filename}`,
          name: f.filename.replace(".wav", ""),
          timestamp: Date.now() - idx * 1e3
          // Sort by order received (newest first)
        })).sort((a, b) => b.timestamp - a.timestamp);
        console.log("Loaded outputs:", qwen3Outputs);
        setOutputs(qwen3Outputs);
      } catch (err) {
        console.error("Failed to load outputs:", err);
      }
    }, []);
    React2.useEffect(() => {
      loadOutputs();
      const interval = setInterval(loadOutputs, 1e4);
      return () => clearInterval(interval);
    }, [loadOutputs]);
    React2.useEffect(() => {
      const fetchVoices = async () => {
        try {
          const res = await fetch("/api/apps/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ app_name: appName, params: { mode: "list" } })
          });
          const data = await res.json();
          if (data.task_id) {
            const poll = setInterval(async () => {
              const statusRes = await fetch(`/status/${data.task_id}`);
              const statusData = await statusRes.json();
              if (statusData.status === "completed") {
                clearInterval(poll);
                const voiceData = statusData.output?.voices || {};
                if (Object.keys(voiceData).length > 0) {
                  setVoices((prev) => ({ ...prev, ...voiceData }));
                }
              } else if (statusData.status === "failed") {
                clearInterval(poll);
              }
            }, 500);
          }
        } catch (err) {
          console.log("Using default voices");
        }
      };
      fetchVoices();
    }, [appName]);
    React2.useEffect(() => {
      if (!taskId) return;
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`/status/${taskId}`);
          const data = await res.json();
          if (data.status === "completed") {
            setIsGenerating(false);
            const outputPath = typeof data.output === "object" ? data.output["1"] : data.output;
            if (outputPath && updateTask) {
              loadOutputs();
              updateTask(taskId, { status: "completed", output: { "1": outputPath }, progress: 100 });
              addToast({ message: "Audio generated!", type: "success" });
            }
            setTaskId(null);
          } else if (data.status === "failed") {
            setIsGenerating(false);
            if (updateTask) updateTask(taskId, { status: "failed", errorMessage: data.error || "Failed" });
            setTaskId(null);
            addToast({ message: "Generation failed", type: "error" });
          } else if (updateTask) {
            updateTask(taskId, { status: data.status, progress: data.progress || 0, message: data.message || data.stage });
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 1e3);
      return () => clearInterval(interval);
    }, [taskId, updateTask, addToast, loadOutputs]);
    const handleGenerate = async () => {
      if (!text.trim()) {
        addToast({ message: "Please enter some text", type: "warning" });
        return;
      }
      setIsGenerating(true);
      try {
        const res = await fetch("/api/apps/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            app_name: appName,
            params: { mode: "generate", text: text.trim(), ref_audio_id: selectedVoice, language }
          })
        });
        const data = await res.json();
        if (data.task_id) {
          setTaskId(data.task_id);
          if (addTask) addTask({ id: data.task_id, name: `Qwen3 TTS - ${selectedVoice.replace(".wav", "")}`, status: "queued", progress: 0 });
        }
      } catch (err) {
        setIsGenerating(false);
        addToast({ message: "Failed to start generation", type: "error" });
      }
    };
    const handlePlayVoice = async (voiceId) => {
      const voiceData = voices[voiceId];
      if (!voiceData) return;
      if (playingVoice === voiceId) {
        voiceAudioRef.current.pause();
        voiceAudioRef.current.currentTime = 0;
        setPlayingVoice(null);
        return;
      }
      voiceAudioRef.current.pause();
      setPlayingVoice(null);
      setDownloadingVoice(voiceId);
      try {
        const res = await fetch("/api/apps/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            app_name: appName,
            params: { mode: "get_voice_audio", audio_id: voiceId }
          })
        });
        const data = await res.json();
        if (data.task_id) {
          const audioResult = await new Promise((resolve, reject) => {
            const poll = setInterval(async () => {
              const statusRes = await fetch(`/status/${data.task_id}`);
              const statusData = await statusRes.json();
              if (statusData.status === "completed") {
                clearInterval(poll);
                resolve(statusData.output);
              } else if (statusData.status === "failed") {
                clearInterval(poll);
                reject(new Error(statusData.error || "Failed to get voice audio"));
              }
            }, 500);
          });
          if (audioResult.filepath) {
            setVoices((prev) => ({
              ...prev,
              [voiceId]: { ...prev[voiceId], filepath: audioResult.filepath }
            }));
          }
          const audioUrl = `data:${audioResult.mime_type};base64,${audioResult.audio_base64}`;
          voiceAudioRef.current.src = audioUrl;
          voiceAudioRef.current.onended = () => setPlayingVoice(null);
          voiceAudioRef.current.onerror = (e) => {
            console.error("Audio error:", e);
            addToast({ message: "Failed to play voice", type: "error" });
            setPlayingVoice(null);
          };
          await voiceAudioRef.current.play();
          setPlayingVoice(voiceId);
        }
      } catch (err) {
        console.error("Voice playback error:", err);
        addToast({ message: "Failed to play voice: " + err.message, type: "error" });
      } finally {
        setDownloadingVoice(null);
      }
    };
    const handleUploadVoice = React2.useCallback(async ({ name, description, referenceText, file }) => {
      const audioId = name.toLowerCase().replace(/\s+/g, "_");
      const audioIdWithExt = audioId + ".wav";
      if (voices[audioIdWithExt]) {
        addToast({ message: "Voice already exists", type: "error" });
        return false;
      }
      try {
        const formData = new FormData();
        formData.append("file", file);
        const uploadRes = await fetch("/api/upload", { method: "POST", body: formData });
        if (!uploadRes.ok) throw new Error("Upload failed");
        const uploadData = await uploadRes.json();
        const uploadedPath = uploadData.file_path || uploadData.path || uploadData.filename;
        const res = await fetch("/api/apps/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            app_name: appName,
            params: { mode: "upload", audio_id: audioId, audio_path: uploadedPath, text: referenceText || "", tags: description, language: "English" }
          })
        });
        const data = await res.json();
        if (data.task_id) {
          const checkStatus = setInterval(async () => {
            const statusRes = await fetch(`/status/${data.task_id}`);
            const statusData = await statusRes.json();
            if (statusData.status === "completed") {
              clearInterval(checkStatus);
              setVoices((prev) => ({ ...prev, [audioIdWithExt]: { text: referenceText || "", tags: description, language: "English", url: "", filepath: `/root/exiv-private/apps/qwen3_tts/wavs/${audioId}.wav` } }));
              addToast({ message: "Voice added!", type: "success" });
            } else if (statusData.status === "failed") {
              clearInterval(checkStatus);
            }
          }, 1e3);
          return true;
        }
      } catch (err) {
        addToast({ message: "Failed to add voice: " + err.message, type: "error" });
        return false;
      }
    }, [appName, voices, addToast]);
    const filteredVoices = Object.entries(voices).filter(([id, data]) => {
      const searchLower = searchQuery.toLowerCase();
      return id.toLowerCase().includes(searchLower) || data.tags?.toLowerCase().includes(searchLower);
    });
    const selectedVoiceData = voices[selectedVoice];
    return /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-container" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-left-panel" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-panel-header" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-header-icon" }, /* @__PURE__ */ React2.createElement(WaveformIcon, null)), /* @__PURE__ */ React2.createElement("h2", { className: "qwen3-tts-panel-title" }, "Qwen3 Audio")), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-search-row" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-search-container" }, /* @__PURE__ */ React2.createElement(SearchIcon, null), /* @__PURE__ */ React2.createElement("input", { type: "text", placeholder: "Search voices...", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value), className: "qwen3-tts-search-input" })), /* @__PURE__ */ React2.createElement("button", { className: "qwen3-tts-add-btn", onClick: () => setIsUploadModalOpen(true), title: "Add new voice" }, /* @__PURE__ */ React2.createElement(PlusIcon, null))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-voice-list" }, filteredVoices.map(([voiceId, voiceData]) => /* @__PURE__ */ React2.createElement(VoiceCard, { key: voiceId, voiceId, voiceData, isSelected: selectedVoice === voiceId, isPlaying: playingVoice === voiceId, isLoading: downloadingVoice === voiceId, onSelect: () => setSelectedVoice(voiceId), onPlay: handlePlayVoice }))), selectedVoiceData && /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-reference-panel" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-reference-label" }, "Reference Text ", !selectedVoiceData.text && /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-xvector-badge-inline" }, "X-Vector")), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-reference-text" }, selectedVoiceData.text || "No reference text - x-vector mode"))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-center-panel" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-center-header" }, /* @__PURE__ */ React2.createElement("h1", { className: "qwen3-tts-title" }, "Text to Speech"), /* @__PURE__ */ React2.createElement("p", { className: "qwen3-tts-subtitle" }, "Generate natural speech with voice cloning")), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-controls-row" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-control-group" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-control-label" }, "Language"), /* @__PURE__ */ React2.createElement("select", { value: language, onChange: (e) => setLanguage(e.target.value), className: "qwen3-tts-select" }, LANGUAGES.map((lang) => /* @__PURE__ */ React2.createElement("option", { key: lang, value: lang }, lang)))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-control-group" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-control-label" }, "Selected Voice"), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-selected-voice-display" }, /* @__PURE__ */ React2.createElement(MicIcon, null), /* @__PURE__ */ React2.createElement("span", null, selectedVoice.replace(".wav", "").replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())), !selectedVoiceData?.text && /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-xvector-badge-inline" }, "X-Vector")))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-input-section" }, /* @__PURE__ */ React2.createElement("label", { className: "qwen3-tts-input-label" }, "Text to Synthesize"), /* @__PURE__ */ React2.createElement("textarea", { value: text, onChange: (e) => setText(e.target.value), placeholder: "Enter the text you want to convert to speech...", className: "qwen3-tts-textarea" }), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-input-footer" }, /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-char-count" }, text.length, " characters"), /* @__PURE__ */ React2.createElement("button", { onClick: handleGenerate, disabled: isGenerating || !text.trim(), className: `qwen3-tts-generate-btn ${isGenerating ? "qwen3-tts-generate-btn-loading" : ""}` }, isGenerating ? /* @__PURE__ */ React2.createElement(React2.Fragment, null, /* @__PURE__ */ React2.createElement("span", { className: "qwen3-tts-spinner" }), /* @__PURE__ */ React2.createElement("span", null, "Generating...")) : /* @__PURE__ */ React2.createElement(React2.Fragment, null, /* @__PURE__ */ React2.createElement(SparklesIcon, null), /* @__PURE__ */ React2.createElement("span", null, "Generate Speech")))))), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-right-panel" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-panel-header" }, /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-header-icon" }, /* @__PURE__ */ React2.createElement(VolumeIcon, null)), /* @__PURE__ */ React2.createElement("h2", { className: "qwen3-tts-panel-title" }, "Generated Audio (", outputs.length, ")")), /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-outputs-list" }, outputs.length === 0 ? /* @__PURE__ */ React2.createElement("div", { className: "qwen3-tts-outputs-empty" }, "No generated audio yet") : outputs.map((output) => /* @__PURE__ */ React2.createElement(OutputItem, { key: output.id, output })))), /* @__PURE__ */ React2.createElement(UploadModal, { isOpen: isUploadModalOpen, onClose: () => setIsUploadModalOpen(false), onUpload: handleUploadVoice }));
  };
  window.ExivPlugins = window.ExivPlugins || {};
  window.ExivPlugins["Qwen3 TTS"] = App;
  console.log("Qwen3 TTS Plugin Registered!");
})(React);
