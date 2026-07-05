'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody } from '../../components/ui/Card';
import { ProgressBar } from '../../components/ui/ProgressBar';
import styles from './page.module.css';

const STAGES = [
  { id: 'validation', label: 'Validation' },
  { id: 'preprocessing', label: 'Preprocessing' },
  { id: 'segmentation', label: 'Segmentation' },
  { id: 'roi', label: 'ROI Extraction' },
  { id: 'classification', label: 'Classification' },
  { id: 'explainability', label: 'Explainability (Grad-CAM)' },
  { id: 'report', label: 'Report Generation' },
];

export default function InferenceMonitor() {
  const [progress, setProgress] = useState<Record<string, number>>({});
  
  // Simulate progress for demonstration
  useEffect(() => {
    let currentStageIndex = 0;
    
    const interval = setInterval(() => {
      setProgress(prev => {
        const stage = STAGES[currentStageIndex];
        if (!stage) {
          clearInterval(interval);
          return prev;
        }

        const currentVal = prev[stage.id] || 0;
        if (currentVal >= 100) {
          currentStageIndex++;
          return prev;
        }
        
        return {
          ...prev,
          [stage.id]: Math.min(currentVal + Math.random() * 20, 100)
        };
      });
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Inference Monitor</h1>
        <p className={styles.subtitle}>Study #8492 - Live Execution</p>
      </header>

      <div className={styles.content}>
        <Card>
          <CardHeader title="Execution Stages" />
          <CardBody>
            <div className={styles.stages}>
              {STAGES.map((stage) => {
                const val = progress[stage.id] || 0;
                let status: 'default' | 'success' | 'warning' = 'default';
                if (val === 100) status = 'success';
                else if (val > 0) status = 'warning';

                return (
                  <div key={stage.id} className={styles.stageRow}>
                    <div className={styles.stageInfo}>
                      <span className={styles.stageLabel}>{stage.label}</span>
                      <span className={styles.stageStatus}>
                        {val === 100 ? 'Completed' : val > 0 ? 'In Progress' : 'Pending'}
                      </span>
                    </div>
                    <ProgressBar 
                      value={val} 
                      status={status}
                      showValue={true} 
                    />
                  </div>
                );
              })}
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
