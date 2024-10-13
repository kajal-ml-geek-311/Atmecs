import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import Diagnosis from './pages/Diagnosis';
import SyntheticReport from './pages/SyntheticReport';
import TreatmentPlan from './pages/TreatmentPlan';
import Chatbot from './pages/chatbot';
import Home from './pages/Home';
import logo from './assets/logo.png';

const App: React.FC = () => {
  const { t, i18n } = useTranslation();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };

  return (
    <Router>
      <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-100 to-purple-100 font-sans">
        <div 
          className="absolute inset-0 bg-cover bg-center z-0" 
          style={{
            backgroundImage: "url('https://img.freepik.com/free-vector/abstract-medical-background-with-hexagons-pattern_1017-26877.jpg')",
            opacity: 0.1
          }}
        ></div>
        <div className="relative z-10">
          <nav className="bg-white bg-opacity-90 shadow-md backdrop-blur-md">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex items-center">
                  <Link to="/" className="flex items-center">
                    <img src={logo} alt="MediAssist AI Logo" className="h-8 w-auto mr-2" style={{ maxWidth: '32px' }} />
                    <span className="text-xl font-semibold text-blue-600">{t('app_name')}</span>
                  </Link>
                </div>
                <div className="flex items-center">
                  <Link to="/diagnosis" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">{t('diagnosis')}</Link>
                  <Link to="/synthetic-report" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">{t('synthetic_report')}</Link>
                  <Link to="/treatment-plan" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">{t('treatment_plan')}</Link>
                  <Link to="/chatbot" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">{t('MediAssist Bot')}</Link>
                  <select
                    onChange={(e) => changeLanguage(e.target.value)}
                    value={i18n.language}
                    className="ml-4 bg-white border border-gray-300 rounded-md text-gray-700 text-sm"
                  >
                    <option value="en">English</option>
                    <option value="hi">हिंदी</option>
                  </select>
                </div>
              </div>
            </div>
          </nav>

          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/diagnosis" element={<Diagnosis />} />
              <Route path="/synthetic-report" element={<SyntheticReport />} />
              <Route path="/treatment-plan" element={<TreatmentPlan />} />
              <Route path="/chatbot" element={<Chatbot />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;