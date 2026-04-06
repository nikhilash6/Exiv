import React from 'react';
import App from './App.jsx';

// Ensure the global registry exists
window.ExivPlugins = window.ExivPlugins || {};

// Register the component using the exact name of the Python App
window.ExivPlugins['Qwen3 TTS'] = App;
console.log("Qwen3 TTS Plugin Registered!");
