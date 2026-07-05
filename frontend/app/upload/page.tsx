'use client';

import React, { useState } from 'react';
import { Card, CardBody } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { UploadCloud, File, X } from 'lucide-react';
import styles from './page.module.css';

export default function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Upload MRI Study</h1>
        <p className={styles.subtitle}>Supported formats: DICOM, NIfTI (.nii, .nii.gz)</p>
      </header>

      <Card>
        <CardBody>
          <div 
            className={`${styles.dropZone} ${isDragging ? styles.dragging : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {file ? (
              <div className={styles.fileInfo}>
                <File size={48} className={styles.fileIcon} />
                <div className={styles.fileDetails}>
                  <div className={styles.fileName}>{file.name}</div>
                  <div className={styles.fileSize}>{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                </div>
                <button className={styles.removeButton} onClick={() => setFile(null)}>
                  <X size={20} />
                </button>
              </div>
            ) : (
              <div className={styles.uploadPrompt}>
                <UploadCloud size={48} className={styles.uploadIcon} />
                <h3>Drag & Drop Study Files</h3>
                <p>or click to browse from your computer</p>
                <div className={styles.uploadActions}>
                  <Button variant="primary">Browse Files</Button>
                </div>
              </div>
            )}
          </div>

          {file && (
            <div className={styles.actions}>
              <Button variant="outline" onClick={() => setFile(null)}>Cancel</Button>
              <Button variant="primary">Start Inference</Button>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
