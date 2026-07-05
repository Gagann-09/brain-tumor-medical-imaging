import React from 'react';
import clsx from 'clsx';
import styles from './ProgressBar.module.css';

interface ProgressBarProps {
  value: number; // 0 to 100
  label?: string;
  status?: 'default' | 'success' | 'error' | 'warning';
  showValue?: boolean;
  className?: string;
}

export const ProgressBar = ({ 
  value, 
  label, 
  status = 'default', 
  showValue = true,
  className 
}: ProgressBarProps) => {
  const clampedValue = Math.min(Math.max(value, 0), 100);

  return (
    <div className={clsx(styles.container, className)}>
      {(label || showValue) && (
        <div className={styles.labelContainer}>
          {label && <span className={styles.label}>{label}</span>}
          {showValue && <span className={styles.value}>{Math.round(clampedValue)}%</span>}
        </div>
      )}
      <div className={styles.track}>
        <div 
          className={clsx(styles.fill, status !== 'default' && styles[status])}
          style={{ width: `${clampedValue}%` }}
        />
      </div>
    </div>
  );
};
