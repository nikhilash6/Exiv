# Building Custom App UI

Exiv Apps can define complex inputs that are automatically parsed into User Interfaces if you are running the API Server with a corresponding Frontend. However, you can also build a completely custom UI plugin.

## Standard Input Types

When defining an `App` class, you define `Input` objects. These dictate how a generic frontend UI will be rendered.

Example UI Inputs:

```python
from exiv.utils.inputs import Input

my_app = App(
    name="Generator",
    inputs={
        'prompt': Input(label="Text Prompt", type="string"),
        'reference_image': Input(label="Reference", type="image"),
        'strength': Input(label="Strength", type="float", default=0.75),
    },
    # ...
)
```

By standardizing inputs this way, the Exiv API server can tell any compatible UI exactly what form fields are required to run the App successfully.

## Building a Custom UI Plugin

A custom UI plugin requires building essential components, like hooking into the task queueing API to run the app and the output fetching API to display results.

The Exiv server only picks up a compiled `/dist/index.js` and `/dist/style.css` from an `apps/<app_name>/ui/` directory.

### 1. Project Setup

Inside your app's directory (e.g., `apps/calculator/`), create a new React project using Vite:

```bash
cd apps/calculator
npm create vite@latest ui -- --template react
cd ui
npm install
```

### 2. Vite Configuration

To ensure your app runs correctly when dynamically loaded by the main Exiv frontend, you must configure Vite to:
- Output to the `../dist` folder without hashes in the filenames (`index.js`, `style.css`).
- Build an IIFE (Immediately Invoked Function Expression).
- Use the classic JSX runtime and externalize React so it uses the globally provided `window.React`.

Update your `vite.config.js`:

```javascript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // Use classic JSX runtime to bind correctly to the global React object
  plugins: [react({ jsxRuntime: 'classic' })],
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    minify: false, // Optional: helpful for debugging
    lib: {
      entry: 'src/main.jsx',
      name: 'MyAppPlugin',
      formats: ['iife'],
      fileName: () => 'index.js'
    },
    rollupOptions: {
      // Do not bundle React, rely on the host environment
      external: ['react', 'react-dom'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM'
        },
        assetFileNames: (assetInfo) => {
          if (assetInfo.name === 'style.css') return 'style.css';
          return assetInfo.name;
        }
      }
    }
  },
  define: {
    'process.env.NODE_ENV': '"production"'
  }
});
```

### 3. Registering the Plugin

Instead of rendering to a DOM node using `ReactDOM.createRoot`, your `main.jsx` needs to attach your root React component to the `window.ExivPlugins` registry.

`src/main.jsx`:

```javascript
import React from 'react';
import App from './App.jsx';

// Ensure the global registry exists
window.ExivPlugins = window.ExivPlugins || {};

// Register your component using the exact name of your Python App
window.ExivPlugins['calculator'] = App;
console.log("Calculator Plugin Registered!");
```

### 4. API Integration (Task Queueing & Status)

Inside your main component (`App.jsx`), you will need to interact with the backend to trigger processing and poll for updates.

Exiv exposes the following endpoints:
- `POST /api/apps/run`: Submits a new task.
- `GET /status/{task_id}`: Retrieves the status of a queued or running task.

You have to hook into the host UI's task drawer by accessing `window.useTaskActions()`. You can also trigger toast notifications using `window.useToast()`.

Example `App.jsx` Component:

```jsx
import React, { useState, useEffect } from 'react';

const MyAppUI = ({ appName = "calculator" }) => {
  const [taskId, setTaskId] = useState(null);
  const [result, setResult] = useState(null);

  // Access Exiv's global task drawer actions
  const actions = window.useTaskActions ? window.useTaskActions() : null;
  const addTask = actions ? actions.addTask : null;

  // Access Exiv's global toast API
  const { addToast } = window.useToast ? window.useToast() : { addToast: (msg) => console.log(msg) };

  // Polling for task status
  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      const res = await fetch(`/status/${taskId}`);
      const data = await res.json();

      if (data.status === 'completed') {
        // Output format depends on your Python App's returned dictionary
        setResult(data.output["1"]);
        setTaskId(null);
        addToast({ message: "Task Completed Successfully!", type: "success" });
      } else if (data.status === 'failed') {
        addToast({ message: "Task Failed", type: "error" });
        setTaskId(null);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId]);

  const runTask = async () => {
    // 1. Submit the task
    const res = await fetch('/api/apps/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        app_name: appName,
        params: {
          // Your Python App Inputs here
          num1: 5,
          num2: 10
        }
      })
    });

    const data = await res.json();
    setTaskId(data.task_id);

    // 2. Register it in the host UI's Task Queue Drawer
    if (addTask) {
      addTask({
        id: data.task_id,
        name: `Calculator Task`,
        status: 'queued',
        progress: 0
      });
    }
  };

  return (
    <div>
      <button onClick={runTask} disabled={!!taskId}>
        {taskId ? 'Running...' : 'Run App'}
      </button>
      {result && <div>Result: {result}</div>}
    </div>
  );
};

export default MyAppUI;
```

### 5. Development Tips

Since the host Exiv frontend dynamically fetches the compiled `index.js` and `style.css` from the `dist` folder on page load, you do not need a separate dev server.

You can simply build your code in watch mode:

```bash
npm run build -- --watch
# OR
npx vite build --watch
```

This will automatically recompile your plugin bundle whenever you save a React file. Then, just refresh the main Exiv browser page to see your latest changes!