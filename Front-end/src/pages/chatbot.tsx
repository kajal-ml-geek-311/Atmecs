import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { useTranslation } from 'react-i18next'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function Chatbot() {
  const { t, i18n } = useTranslation()
  const [patientId, setPatientId] = useState('')
  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const chatContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [conversation])

  const handleSendMessage = async () => {
    if (!patientId || !message) {
      setError(t('fill_required_fields'))
      return
    }

    setIsLoading(true)
    setError('')

    try {
      const payload = {
        patient_id: patientId,
        message: message,
        language: i18n.language,
      }

      const res = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/chatbot`, payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      })

      setConversation([...conversation, { role: 'user', content: message }, { role: 'assistant', content: res.data.response }])
      setMessage('')
    } catch (err: any) {
      console.error('Chatbot Error:', err)
      setError(t('error_generating_response'))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 p-8 flex items-center justify-center">
      <div className="w-full max-w-2xl bg-white bg-opacity-80 backdrop-filter backdrop-blur-lg rounded-xl shadow-xl overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6">
          <h2 className="text-2xl font-bold">{t('ai_doctor_chatbot')}</h2>
        </div>
        <div className="p-6 space-y-6">
          <div>
            <label htmlFor="patientId" className="block text-sm font-medium text-gray-700 mb-1">
              {t('patient_id')}
            </label>
            <input
              type="text"
              id="patientId"
              value={patientId}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPatientId(e.target.value)}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              placeholder={t('enter_your_patient_id')}
              required
            />
          </div>

          <div 
            ref={chatContainerRef}
            className="h-[400px] overflow-y-auto rounded-md border p-4 space-y-4"
          >
            {conversation.map((msg, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg ${
                  msg.role === 'assistant'
                    ? 'bg-blue-100 text-blue-800 ml-4'
                    : 'bg-purple-100 text-purple-800 mr-4'
                }`}
              >
                <strong className="block mb-1">
                  {msg.role === 'assistant' ? t('assistant') : t('you')}:
                </strong>
                {msg.content}
              </div>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={message}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMessage(e.target.value)}
              className="flex-grow rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              placeholder={t('type_your_message')}
              onKeyPress={(e: React.KeyboardEvent) => e.key === 'Enter' && handleSendMessage()}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 transition-colors duration-200"
            >
              {isLoading ? t('sending') : t('send')}
            </button>
          </div>

          {error && (
            <p className="text-red-600">
              {error}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}