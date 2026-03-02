import React from 'react';
import App from './App.jsx';

window.ExivPlugins = window.ExivPlugins || {};
window.ExivPlugins['Frame Interpolation'] = App;
console.log("Frame Interpolation Plugin Registered!");