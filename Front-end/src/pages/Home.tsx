import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const Home: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="container mx-auto p-4">
      <div className="flex flex-col items-center">
        <div className="bg-white bg-opacity-80 shadow-xl rounded-lg p-8 backdrop-blur-md w-full max-w-4xl">
          <h1 className="text-4xl font-bold mb-6 text-blue-600">{t('welcome_to_mediassist')}</h1>
          <p className="mb-8 text-gray-700 text-lg">
            {t('mediassist_description')}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Link to="/diagnosis" className="bg-gradient-to-r from-blue-500 to-blue-600 p-6 rounded-lg hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <h2 className="text-2xl font-semibold mb-3 text-white">{t('diagnosis')}</h2>
              <p className="text-blue-100">{t('diagnosis_description')}</p>
            </Link>
            <Link to="/synthetic-report" className="bg-gradient-to-r from-green-500 to-green-600 p-6 rounded-lg hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <h2 className="text-2xl font-semibold mb-3 text-white">{t('synthetic_report')}</h2>
              <p className="text-green-100">{t('synthetic_report_description')}</p>
            </Link>
            <Link to="/treatment-plan" className="bg-gradient-to-r from-purple-500 to-purple-600 p-6 rounded-lg hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <h2 className="text-2xl font-semibold mb-3 text-white">{t('treatment_plan')}</h2>
              <p className="text-purple-100">{t('treatment_plan_description')}</p>
            </Link>
            <Link to="/chatbot" className="bg-gradient-to-r from-red-500 to-red-600 p-6 rounded-lg hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1">
              <h2 className="text-2xl font-semibold mb-3 text-white">{t('MediAssist Bot')}</h2>
              <p className="text-red-100">{t('Talk to our Medical expert bot')}</p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;