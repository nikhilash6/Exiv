import React from 'react';
import App from './App.jsx';

window.ExivPlugins = window.ExivPlugins || {};
window.ExivPlugins['Character Replace'] = App;
console.log("Character Replace Plugin Registered!");
