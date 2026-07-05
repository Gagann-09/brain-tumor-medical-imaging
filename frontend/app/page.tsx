'use client';

import React from 'react';
import { Card, CardHeader, CardBody } from '../components/ui/Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import styles from './page.module.css';

const performanceData = [
  { time: '08:00', latency: 120 },
  { time: '09:00', latency: 140 },
  { time: '10:00', latency: 110 },
  { time: '11:00', latency: 160 },
  { time: '12:00', latency: 130 },
  { time: '13:00', latency: 125 },
];

export default function Dashboard() {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>System Dashboard</h1>
        <p className={styles.subtitle}>Operational status and inference queue</p>
      </header>

      <div className={styles.grid}>
        {/* System Status Section */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>System Status</h2>
          <div className={styles.statusCards}>
            <Card>
              <CardBody>
                <div className={styles.metricLabel}>API Health</div>
                <div className={styles.metricValue} style={{ color: 'var(--color-success)' }}>Operational</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody>
                <div className={styles.metricLabel}>GPU Utilization</div>
                <div className={styles.metricValue}>45%</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody>
                <div className={styles.metricLabel}>Storage Available</div>
                <div className={styles.metricValue}>2.4 TB</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody>
                <div className={styles.metricLabel}>Active Workers</div>
                <div className={styles.metricValue}>8 / 8</div>
              </CardBody>
            </Card>
          </div>
        </section>

        {/* Inference Queue & Performance */}
        <div className={styles.twoColumn}>
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Inference Queue</h2>
            <Card className={styles.fullHeight}>
              <CardBody>
                <ul className={styles.list}>
                  <li className={styles.listItem}>
                    <div>
                      <div className={styles.listTitle}>Study #8492</div>
                      <div className={styles.listSub}>Preprocessing...</div>
                    </div>
                    <span className={styles.badgeWarning}>In Progress</span>
                  </li>
                  <li className={styles.listItem}>
                    <div>
                      <div className={styles.listTitle}>Study #8491</div>
                      <div className={styles.listSub}>Queued</div>
                    </div>
                    <span className={styles.badgeNeutral}>Waiting</span>
                  </li>
                </ul>
              </CardBody>
            </Card>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Performance (ms)</h2>
            <Card className={styles.fullHeight}>
              <CardBody>
                <div style={{ width: '100%', height: 250 }}>
                  <ResponsiveContainer>
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-primary-200)" />
                      <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={12} />
                      <YAxis stroke="var(--text-muted)" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'var(--bg-surface)', 
                          border: 'var(--border-light)',
                          borderRadius: 'var(--radius-md)'
                        }} 
                      />
                      <Line type="monotone" dataKey="latency" stroke="var(--color-secondary-500)" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardBody>
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}
