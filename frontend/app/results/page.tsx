'use client';

import React, { useState } from 'react';
import { MRIViewer } from '../../components/features/MRIViewer';
import { Card, CardHeader, CardBody } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Download, FileText, Share2 } from 'lucide-react';
import styles from './page.module.css';

// Mock data for layers
const initialLayers = [
  { id: 'mri', name: 'Original MRI (T1ce)', visible: true, opacity: 1, type: 'base' as const },
  { id: 'mask', name: 'Segmentation Mask', visible: true, opacity: 0.4, type: 'mask' as const },
  { id: 'tumor', name: 'Tumor Boundary', visible: false, opacity: 1, type: 'boundary' as const },
  { id: 'gradcam', name: 'Grad-CAM', visible: false, opacity: 0.6, type: 'heatmap' as const },
];

export default function ResultsWorkspace() {
  const [layers, setLayers] = useState(initialLayers);

  return (
    <div className={styles.workspace}>
      <div className={styles.main}>
        <div className={styles.viewerContainer}>
          <MRIViewer 
            layers={layers} 
            onLayerChange={setLayers}
            metadata={{
              'Study ID': 'MRI-8492',
              'Patient': 'P-1042',
              'Modality': 'T1ce',
              'Resolution': '256x256'
            }}
          />
        </div>
      </div>
      
      <aside className={styles.contextPanel}>
        <div className={styles.panelSection}>
          <h2 className={styles.panelTitle}>Clinical Summary</h2>
          <Card>
            <CardBody>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Classification</span>
                <span className={styles.summaryValue} style={{ color: 'var(--color-error)' }}>Glioma</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Confidence</span>
                <span className={styles.summaryValue}>98.7%</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Tumor Volume</span>
                <span className={styles.summaryValue}>14.2 cm³</span>
              </div>
            </CardBody>
          </Card>
        </div>

        <div className={styles.panelSection}>
          <h2 className={styles.panelTitle}>Actions</h2>
          <div className={styles.actionGrid}>
            <Button variant="outline" icon={<FileText size={16} />}>Generate Report</Button>
            <Button variant="outline" icon={<Download size={16} />}>Export DICOM</Button>
            <Button variant="outline" icon={<Share2 size={16} />}>Share Study</Button>
          </div>
        </div>
      </aside>
    </div>
  );
}
