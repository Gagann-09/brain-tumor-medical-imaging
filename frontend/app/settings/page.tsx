'use client';

import React from 'react';
import { Card, CardHeader, CardBody } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';

export default function SettingsPage() {
  return (
    <div style={{ padding: 'var(--space-8)', maxWidth: 800, margin: '0 auto', width: '100%' }}>
      <header style={{ marginBottom: 'var(--space-8)' }}>
        <h1 style={{ fontSize: '1.5rem', marginBottom: 'var(--space-1)' }}>Settings</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Configure application preferences</p>
      </header>

      <Card>
        <CardHeader title="Appearance" />
        <CardBody>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
            <p>Theme: <strong>Medical Light</strong> (Dark mode coming soon)</p>
            <p>Default Layout: <strong>Standard</strong></p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
