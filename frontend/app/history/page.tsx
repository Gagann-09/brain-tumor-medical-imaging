'use client';

import React from 'react';
import { Card, CardBody } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Search, Filter, Eye } from 'lucide-react';
import styles from './page.module.css';

const HISTORY_DATA = [
  { id: 'MRI-8492', patient: 'P-1042', date: '2026-07-05', type: 'Glioma', confidence: 98.7, status: 'Reviewed', version: 'v1.4' },
  { id: 'MRI-8491', patient: 'P-0931', date: '2026-07-04', type: 'Meningioma', confidence: 94.2, status: 'Pending Review', version: 'v1.4' },
  { id: 'MRI-8490', patient: 'P-1122', date: '2026-07-04', type: 'Pituitary', confidence: 96.5, status: 'Reviewed', version: 'v1.4' },
  { id: 'MRI-8489', patient: 'P-0844', date: '2026-07-03', type: 'No Tumor', confidence: 99.1, status: 'Reviewed', version: 'v1.4' },
  { id: 'MRI-8488', patient: 'P-1005', date: '2026-07-02', type: 'Glioma', confidence: 89.4, status: 'Requires Attention', version: 'v1.3' },
];

export default function PatientHistory() {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Patient History</h1>
        <p className={styles.subtitle}>Search and filter past inference results</p>
      </header>

      <div className={styles.toolbar}>
        <div className={styles.searchBox}>
          <Search size={18} className={styles.searchIcon} />
          <input type="text" placeholder="Search by Study ID or Patient..." className={styles.searchInput} />
        </div>
        <Button variant="outline" icon={<Filter size={16} />}>Filter</Button>
      </div>

      <Card>
        <CardBody style={{ padding: 0 }}>
          <div className={styles.tableContainer}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Study ID</th>
                  <th>Patient</th>
                  <th>Date</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Status</th>
                  <th>Model</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {HISTORY_DATA.map(row => (
                  <tr key={row.id}>
                    <td className={styles.mono}>{row.id}</td>
                    <td className={styles.mono}>{row.patient}</td>
                    <td>{row.date}</td>
                    <td>{row.type}</td>
                    <td>{row.confidence}%</td>
                    <td>
                      <span className={`${styles.statusBadge} ${styles[row.status.replace(' ', '')] || styles.default}`}>
                        {row.status}
                      </span>
                    </td>
                    <td className={styles.mono}>{row.version}</td>
                    <td>
                      <Button variant="ghost" size="sm" icon={<Eye size={16} />} title="View Results" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
