'use client';

import React from 'react';
import { Card, CardHeader, CardBody } from '../../components/ui/Card';
import { ProgressBar } from '../../components/ui/ProgressBar';
import styles from './page.module.css';

export default function SystemHealth() {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>System Health</h1>
        <p className={styles.subtitle}>Detailed infrastructure metrics</p>
      </header>

      <div className={styles.grid}>
        <Card>
          <CardHeader title="Core Services" />
          <CardBody>
            <div className={styles.serviceList}>
              <div className={styles.serviceItem}>
                <span>API Gateway (/api/v1)</span>
                <span className={styles.statusSuccess}>Online</span>
              </div>
              <div className={styles.serviceItem}>
                <span>Database (PostgreSQL)</span>
                <span className={styles.statusSuccess}>Online</span>
              </div>
              <div className={styles.serviceItem}>
                <span>Message Queue (Redis)</span>
                <span className={styles.statusSuccess}>Online</span>
              </div>
              <div className={styles.serviceItem}>
                <span>Background Workers (Celery)</span>
                <span className={styles.statusSuccess}>Online</span>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Resource Utilization" />
          <CardBody>
            <div className={styles.resourceList}>
              <ProgressBar value={45} label="GPU 0 (NVIDIA RTX 4090)" status="default" />
              <ProgressBar value={12} label="GPU 1 (NVIDIA RTX 4090)" status="default" />
              <ProgressBar value={68} label="System Memory" status="warning" />
              <ProgressBar value={32} label="Storage" status="default" />
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
