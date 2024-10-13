import React, { useState } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { jsPDF } from 'jspdf';

const Diagnosis: React.FC = () => {
  const { t, i18n } = useTranslation();
  
  // New State for patient_id
  const [patientId, setPatientId] = useState('');
  
  const [image, setImage] = useState<File | null>(null);
  const [patientData, setPatientData] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [audioSrc, setAudioSrc] = useState('');
  const [heatmapSrc, setHeatmapSrc] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [showHeatmap, setShowHeatmap] = useState(true);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!patientId || !image || !patientData) { // Include patientId in validation
      setError(t('fill_required_fields'));
      return;
    }

    setIsLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('patient_id', patientId); // New Field
    formData.append('image', image);
    formData.append('patient_data', patientData);
    formData.append('language', i18n.language);

    try {
      const response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/diagnosis`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setDiagnosis(response.data.diagnosis);
      if (response.data.audio) {
        setAudioSrc(`data:audio/mp3;base64,${response.data.audio}`);
      }
      if (response.data.heatmap) {
        setHeatmapSrc(`data:image/png;base64,${response.data.heatmap}`);
      }
    } catch (error: any) {
      // Enhanced Error Handling
      if (axios.isAxiosError(error) && error.response) {
        console.error('Error data:', error.response.data);
        setError(error.response.data.detail || t('error_occurred'));
      } else {
        console.error('Error:', error);
        setError(t('error_occurred'));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadReport = () => {
    const doc = new jsPDF();
    doc.text(t('diagnosis_result'), 10, 10);
    doc.text(diagnosis, 10, 20);
    if (heatmapSrc) {
      doc.addImage(heatmapSrc, 'PNG', 10, 100, 180, 180);
    }
    doc.save('diagnosis_report.pdf');
  };

  return (
    <div className="bg-white bg-opacity-80 shadow-xl rounded-lg p-8 backdrop-blur-md">
      <h2 className="text-3xl font-semibold mb-6 text-blue-600">{t('diagnosis')}</h2>
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
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            required
          />
        </div>
        
        {/* Existing Fields */}
        <div>
          <label htmlFor="image" className="block text-sm font-medium text-gray-700 mb-2">{t('upload_xray')}</label>
          <input
            type="file"
            id="image"
            accept="image/jpeg,image/png"
            onChange={handleImageChange}
            className="w-full text-sm text-gray-500 border border-gray-300 rounded-lg cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>
        <div>
          <label htmlFor="patientData" className="block text-sm font-medium text-gray-700 mb-2">{t('patient_data')}</label>
          <textarea
            id="patientData"
            value={patientData}
            onChange={(e) => setPatientData(e.target.value)}
            rows={4}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            placeholder={t('patient_data_placeholder')}
            required
          ></textarea>
        </div>
        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 transition duration-150 ease-in-out"
        >
          {isLoading ? t('processing') : t('get_diagnosis')}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {diagnosis && (
        <div className="mt-8 bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-medium text-blue-800 mb-4">{t('diagnosis_result')}</h3>
          <p className="text-gray-700">{diagnosis}</p>
          {audioSrc && (
            <div className="mt-6">
              <h4 className="text-lg font-medium text-blue-800 mb-2">{t('audio_diagnosis')}</h4>
              <audio controls src={audioSrc} className="w-full">
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
          {heatmapSrc && (
            <div className="mt-6">
              <h4 className="text-lg font-medium text-blue-800 mb-2">{t('heatmap')}</h4>
              <div className="flex items-center mb-2">
                <label htmlFor="opacity" className="mr-2">{t('adjust_transparency')}</label>
                <input
                  type="range"
                  id="opacity"
                  min="0"
                  max="1"
                  step="0.1"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <button
                onClick={() => setShowHeatmap(!showHeatmap)}
                className="mb-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                {showHeatmap ? t('hide_heatmap') : t('show_heatmap')}
              </button>
              <TransformWrapper>
                <TransformComponent>
                  <div className="relative">
                    <img src={URL.createObjectURL(image as Blob)} alt="Original X-ray" className="w-full" />
                    {showHeatmap && (
                      <img
                        src={heatmapSrc}
                        alt="Heatmap"
                        className="absolute top-0 left-0 w-full h-full object-cover"
                        style={{ opacity: heatmapOpacity, mixBlendMode: 'multiply' }}
                      />
                    )}
                  </div>
                </TransformComponent>
              </TransformWrapper>
            </div>
          )}
          <button
            onClick={handleDownloadReport}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            {t('download_report')}
          </button>
        </div>
      )}
    </div>
  );
};

export default Diagnosis;
