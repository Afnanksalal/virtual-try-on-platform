"use client";

import { useState } from "react";

export type TabType = 'personal' | 'garment' | 'results' | 'recommendations';

interface TabNavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  showRecommendations?: boolean;
}

interface Tab {
  id: TabType;
  label: string;
  ariaLabel: string;
}

const tabs: Tab[] = [
  { id: 'personal', label: 'Personal Image', ariaLabel: 'Personal Image tab' },
  { id: 'garment', label: 'Garment', ariaLabel: 'Garment tab' },
  { id: 'results', label: 'Try-On Results', ariaLabel: 'Try-On Results tab' },
  { id: 'recommendations', label: 'Recommendations', ariaLabel: 'Recommendations tab' },
];

export default function TabNavigation({ 
  activeTab, 
  onTabChange, 
  showRecommendations = false 
}: TabNavigationProps) {
  const visibleTabs = showRecommendations 
    ? tabs 
    : tabs.filter(tab => tab.id !== 'recommendations');

  return (
    <div className="border-b border-gray-200 bg-white rounded-t-xl">
      {/* Desktop Tab Navigation */}
      <nav 
        className="hidden md:flex space-x-4 lg:space-x-8 px-4 lg:px-6 overflow-x-auto" 
        aria-label="Studio tabs"
        role="tablist"
      >
        {visibleTabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`${tab.id}-panel`}
            id={`${tab.id}-tab`}
            className={`
              py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap transition-colors min-h-[44px]
              ${activeTab === tab.id
                ? 'border-black text-black'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }
            `}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Mobile Dropdown Navigation */}
      <div className="md:hidden px-4 py-3">
        <select
          value={activeTab}
          onChange={(e) => onTabChange(e.target.value as TabType)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm font-medium bg-white focus:outline-none focus:ring-2 focus:ring-black min-h-[44px]"
          aria-label="Select studio tab"
        >
          {visibleTabs.map((tab) => (
            <option key={tab.id} value={tab.id}>
              {tab.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
