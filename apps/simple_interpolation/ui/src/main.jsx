import React from 'react';
import App from './App.jsx';

window.ExivPlugins = window.ExivPlugins || {};
window.ExivPlugins['Simple Interpolation'] = App;
console.log("Simple Interpolation Plugin Registered!");