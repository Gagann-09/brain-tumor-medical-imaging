'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Activity, 
  UploadCloud, 
  MonitorPlay, 
  Layers, 
  History, 
  HeartPulse, 
  Settings,
  Brain
} from 'lucide-react';
import clsx from 'clsx';
import styles from './Sidebar.module.css';

const NAV_ITEMS = [
  { href: '/', label: 'Dashboard', icon: Activity },
  { href: '/upload', label: 'Upload MRI', icon: UploadCloud },
  { href: '/monitor', label: 'Monitor', icon: MonitorPlay },
  { href: '/results', label: 'Results Workspace', icon: Layers },
  { href: '/history', label: 'Patient History', icon: History },
  { href: '/health', label: 'System Health', icon: HeartPulse },
];

export const Sidebar = () => {
  const pathname = usePathname();

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <Brain className={styles.logoIcon} size={32} />
        <span className={styles.logoText}>ARM-GAN</span>
      </div>

      <nav className={styles.nav}>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || (item.href !== '/' && pathname?.startsWith(item.href));
          
          return (
            <Link 
              key={item.href} 
              href={item.href}
              className={clsx(styles.navItem, isActive && styles.active)}
            >
              <Icon className={styles.navIcon} size={24} />
              <span className={styles.navLabel}>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className={styles.spacer} />

      <Link 
        href="/settings"
        className={clsx(styles.navItem, pathname?.startsWith('/settings') && styles.active)}
      >
        <Settings className={styles.navIcon} size={24} />
        <span className={styles.navLabel}>Settings</span>
      </Link>
    </aside>
  );
};
