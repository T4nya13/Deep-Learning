import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import './styles/design-system.css';     // Variables first
import './styles/templates.css';         // Templates second  
import './styles/components.css';        // Components third
import './styles/style.css';            // Main styles LAST (highest priority)
import './styles/auth.css';             // Auth specific styles
import Home from './components/Home';
import Auth from './components/Auth';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/auth" element={<Auth />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
