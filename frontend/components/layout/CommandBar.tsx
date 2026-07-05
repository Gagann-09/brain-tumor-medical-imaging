import React from 'react';
import { Bell, Search } from 'lucide-react';
import styles from './CommandBar.module.css';

export const CommandBar = () => {
  return (
    <header className={styles.commandBar}>
      <div className={styles.left}>
        <div className={styles.breadcrumbs}>
          <span>ARM-GAN Platform</span>
          <span className={styles.breadcrumbSeparator}>/</span>
          <span style={{ color: 'var(--text-primary)' }}>Workspace</span>
        </div>
      </div>
      
      <div className={styles.right}>
        <button className={styles.actionButton} aria-label="Search">
          <Search size={20} />
        </button>
        <button className={styles.actionButton} aria-label="Notifications">
          <Bell size={20} />
        </button>
        <div className={styles.profile}>
          <div className={styles.avatar}>Dr</div>
          <span>Dr. Smith</span>
        </div>
      </div>
    </header>
  );
};
