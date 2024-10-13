import React, { useState } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';

const SyntheticReport: React.FC = () => {
  const { t, i18n } = useTranslation();
  
  // New State for patient_id
  const [patientId, setPatientId] = useState('');
  
  const [reportType, setReportType] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [medicalCondition, setMedicalCondition] = useState('');
  const [report, setReport] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Include patientId in the validation
    if (!patientId || !reportType || !patientAge || !patientGender || !medicalCondition) {
      setError(t('fill_all_fields'));
      return;
    }

    setIsLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('patient_id', patientId); // New Field
    formData.append('report_type', reportType);
    formData.append('patient_age', patientAge);
    formData.append('patient_gender', patientGender);
    formData.append('medical_condition', medicalCondition);
    formData.append('language', i18n.language);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_BACKEND_URL}/generate_report`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setReport(response.data.report);
    } catch (error: any) {
      // Enhanced Error Handling
      if (axios.isAxiosError(error) && error.response) {
        console.error('Error data:', error.response.data);
        setError(error.response.data.detail || t('error_generating_report'));
      } else {
        console.error('Error:', error);
        setError(t('error_generating_report'));
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white bg-opacity-80 shadow-xl rounded-lg p-8 backdrop-blur-md">
      <h2 className="text-3xl font-semibold mb-6 text-green-600">
        {t('synthetic_medical_report_generator')}
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        
        {/* Patient ID Field */}
        <div>
          <label htmlFor="patientId" className="block text-sm font-medium text-gray-700 mb-2">
            {t('patient_id')}
          </label>
          <input
            type="text"
            id="patientId"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-300 focus:ring focus:ring-green-200 focus:ring-opacity-50"
            required
          />
        </div>
        
        {/* Existing Fields */}
        <div>
          <label htmlFor="reportType" className="block text-sm font-medium text-gray-700 mb-2">
            {t('report_type')}
          </label>
          <select
            id="reportType"
            value={reportType}
            onChange={(e) => setReportType(e.target.value)}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-300 focus:ring focus:ring-green-200 focus:ring-opacity-50"
            required
          >
            <option value="">{t('select_report_type')}</option>
            <option value="Radiology Report">{t('radiology_report')}</option>
            <option value="Clinical Note">{t('clinical_note')}</option>
            <option value="Discharge Summary">{t('discharge_summary')}</option>
            <option value="Lab Report">{t('lab_report')}</option>
          </select>
        </div>
        
        <div>
          <label htmlFor="patientAge" className="block text-sm font-medium text-gray-700 mb-2">
            {t('patient_age')}
          </label>
          <input
            type="number"
            id="patientAge"
            value={patientAge}
            onChange={(e) => setPatientAge(e.target.value)}
            min="0"
            max="120"
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-300 focus:ring focus:ring-green-200 focus:ring-opacity-50"
            required
          />
        </div>
        
        <div>
          <label htmlFor="patientGender" className="block text-sm font-medium text-gray-700 mb-2">
            {t('patient_gender')}
          </label>
          <select
            id="patientGender"
            value={patientGender}
            onChange={(e) => setPatientGender(e.target.value)}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-300 focus:ring focus:ring-green-200 focus:ring-opacity-50"
            required
          >
            <option value="">{t('select_gender')}</option>
            <option value="Male">{t('male')}</option>
            <option value="Female">{t('female')}</option>
            <option value="Other">{t('other')}</option>
          </select>
        </div>
        
        <div>
          <label htmlFor="medicalCondition" className="block text-sm font-medium text-gray-700 mb-2">
            {t('medical_condition')}
          </label>
          <input
            type="text"
            id="medicalCondition"
            value={medicalCondition}
            onChange={(e) => setMedicalCondition(e.target.value)}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-300 focus:ring focus:ring-green-200 focus:ring-opacity-50"
            required
          />
        </div>
        
        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 transition duration-150 ease-in-out"
        >
          {isLoading ? t('generating') : t('generate_report')}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {report && (
        <div className="mt-8 bg-green-50 p-6 rounded-lg">
          <h3 className="text-xl font-medium text-green-800 mb-4">
            {t('generated_report')}
          </h3>
          <pre className="whitespace-pre-wrap text-sm text-gray-700">
            {report}
          </pre>
        </div>
      )}
    </div>
  );
};

export default SyntheticReport;
