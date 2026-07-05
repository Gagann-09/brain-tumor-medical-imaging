import React from 'react';
import clsx from 'clsx';
import styles from './Card.module.css';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export const Card = ({ children, className, ...props }: CardProps) => {
  return (
    <div className={clsx(styles.card, className)} {...props}>
      {children}
    </div>
  );
};

export const CardHeader = ({ title, children, className, ...props }: { title?: React.ReactNode; children?: React.ReactNode } & React.HTMLAttributes<HTMLDivElement>) => (
  <div className={clsx(styles.header, className)} {...props}>
    {title && <h3 className={styles.title}>{title}</h3>}
    {children}
  </div>
);

export const CardBody = ({ children, className, ...props }: CardProps) => (
  <div className={clsx(styles.body, className)} {...props}>
    {children}
  </div>
);

export const CardFooter = ({ children, className, ...props }: CardProps) => (
  <div className={clsx(styles.footer, className)} {...props}>
    {children}
  </div>
);
