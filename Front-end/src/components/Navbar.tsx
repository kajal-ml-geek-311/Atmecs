// src/components/Navbar.tsx

import React from 'react';
import { Link } from 'react-router-dom';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-blue-600 p-4">
      <ul className="flex space-x-4 text-white">
        <li>
          <Link to="/diagnosis" className="hover:underline">
            Diagnosis
          </Link>
        </li>
        <li>
          <Link to="/generate-report" className="hover:underline">
            Generate Report
          </Link>
        </li>
        <li>
          <Link to="/treatment-plan" className="hover:underline">
            Treatment Plan
          </Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
