import type { Metadata } from 'next';
import { Inter, Roboto_Mono } from 'next/font/google';
import '../styles/globals.css';
import { Sidebar } from '../components/layout/Sidebar';
import { CommandBar } from '../components/layout/CommandBar';

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const robotoMono = Roboto_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'ARM-GAN | Medical Imaging Workstation',
  description: 'Advanced Radiology Workstation for Brain Tumor Analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${robotoMono.variable}`}>
      <body>
        <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
          <Sidebar />
          <div style={{ 
            flexGrow: 1, 
            marginLeft: '64px',
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            width: 'calc(100% - 64px)'
          }}>
            <CommandBar />
            <main style={{ 
              marginTop: '64px', 
              flexGrow: 1, 
              overflowY: 'auto',
              backgroundColor: 'var(--bg-root)',
              display: 'flex'
            }}>
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
