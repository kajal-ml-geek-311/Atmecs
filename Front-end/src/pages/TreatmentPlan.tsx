import React, { useState } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';

const TreatmentPlan: React.FC = () => {
  const { t, i18n } = useTranslation();
  
  // New State for patient_id
  const [patientId, setPatientId] = useState('');
  
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [medicalCondition, setMedicalCondition] = useState('');
  const [currentMedications, setCurrentMedications] = useState('');
  const [allergies, setAllergies] = useState('');
  const [treatmentPlan, setTreatmentPlan] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Include patientId in the validation
    if (!patientId || !patientName || !patientAge || !patientGender || !medicalCondition) {
      setError(t('fill_required_fields'));
      return;
    }

    setIsLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('patient_id', patientId); // New Field
    formData.append('patient_name', patientName);
    formData.append('patient_age', patientAge);
    formData.append('patient_gender', patientGender);
    formData.append('medical_condition', medicalCondition);
    formData.append('current_medications', currentMedications);
    formData.append('allergies', allergies);
    formData.append('language', i18n.language);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_BACKEND_URL}/treatment_plan`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setTreatmentPlan(response.data.treatment_plan);
    } catch (error: any) {
      // Enhanced Error Handling
      if (axios.isAxiosError(error) && error.response) {
        console.error('Error data:', error.response.data);
        setError(error.response.data.detail || t('error_generating_treatment_plan'));
      } else {
        console.error('Error:', error);
        setError(t('error_generating_treatment_plan'));
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white bg-opacity-80 shadow-xl rounded-lg p-8 backdrop-blur-md">
      <h2 className="text-3xl font-semibold mb-6 text-purple-600">
        {t('personalized_treatment_plan_generator')}
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
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
            required
          />
        </div>
        
        {/* Existing Fields */}
        <div>
          <label htmlFor="patientName" className="block text-sm font-medium text-gray-700 mb-2">
            {t('patient_name')}
          </label>
          <input
            type="text"
            id="patientName"
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
            required
          />
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
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
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
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
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
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
            required
          />
        </div>
        
        <div>
          <label htmlFor="currentMedications" className="block text-sm font-medium text-gray-700 mb-2">
            {t('current_medications')}
          </label>
          <textarea
            id="currentMedications"
            value={currentMedications}
            onChange={(e) => setCurrentMedications(e.target.value)}
            rows={3}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
          ></textarea>
        </div>
        
        <div>
          <label htmlFor="allergies" className="block text-sm font-medium text-gray-700 mb-2">
            {t('allergies')}
          </label>
          <textarea
            id="allergies"
            value={allergies}
            onChange={(e) => setAllergies(e.target.value)}
            rows={3}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-300 focus:ring focus:ring-purple-200 focus:ring-opacity-50"
          ></textarea>
        </div>
        
        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:opacity-50 transition duration-150 ease-in-out"
        >
          {isLoading ? t('generating') : t('generate_treatment_plan')}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {treatmentPlan && (
        <div className="mt-8 bg-purple-50 p-6 rounded-lg">
          <h3 className="text-xl font-medium text-purple-800 mb-4">
            {t('generated_treatment_plan')}
          </h3>
          <pre className="whitespace-pre-wrap text-sm text-gray-700">
            {treatmentPlan}
          </pre>
        </div>
      )}
    </div>
  );
};

export default TreatmentPlan;
