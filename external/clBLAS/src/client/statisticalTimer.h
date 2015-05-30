/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#pragma once
#ifndef _STATISTICALTIMER_H_
#define _STATISTICALTIMER_H_
#include <iosfwd>
#include <vector>
#include <algorithm>

/**
 * \file clAmdFft.StatisticalTimer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 *	This class is implemented entirely in the header, to facilitate inclusion into multiple
 *	projects without needing to compile an object file for each project.
 */

/**
 * \class StatisticalTimer
 * \brief Counter that provides a fairly accurate timing mechanism for both
 * windows and linux. This timer is used extensively in all the samples.
 */

class StatisticalTimer
{
	//	Private typedefs
	typedef std::vector< unsigned long long > clkVector;
	typedef	std::pair< std::string, unsigned int > labelPair;
	typedef	std::vector< labelPair > stringVector;

	//	In order to calculate statistics <std. dev.>, we need to keep a history of our timings
	stringVector	labelID;
	clkVector	clkStart;
	std::vector< clkVector >	clkTicks;

	//	How many clockticks in a second
	unsigned long long	clkFrequency;

	//	Saved sizes for our vectors, used in Reset() to reallocate vectors
	clkVector::size_type	nEvents, nSamples;

	//	This setting controls whether the Timer should convert samples into time by dividing by the
	//	clock frequency
	bool normalize;

	/**
	 * \fn StatisticalTimer()
	 * \brief Constructor for StatisticalTimer that initializes the class
	 *	This is private so that user code cannot create their own instantiation.  Instead, you
	 *	must go through getInstance( ) to get a reference to the class.
	 */
	StatisticalTimer( );

	/**
	 * \fn ~StatisticalTimer()
	 * \brief Destructor for StatisticalTimer that cleans up the class
	 */
	~StatisticalTimer( );

	/**
	 * \fn StatisticalTimer(const StatisticalTimer& )
	 * \brief Copy constructors do not make sense for a singleton, disallow copies
	 */
	StatisticalTimer( const StatisticalTimer& );

	/**
	 * \fn operator=( const StatisticalTimer& )
	 * \brief Assignment operator does not make sense for a singleton, disallow assignments
	 */
	StatisticalTimer& operator=( const StatisticalTimer& );

	friend std::ostream& operator<<( std::ostream& os, const StatisticalTimer& s );

public:
	//	Public typedefs
	typedef stringVector::difference_type sTimerID;

	/**
	 * \fn getInstance()
	 * \brief This returns a reference to the singleton timer.  Guarantees only 1 timer class is ever
	 *	instantiated within a compilable executable.
	 */
	static StatisticalTimer& getInstance( );

	/**
	 * \fn void Start( sTimerID id )
	 * \brief Start the timer
	 * \sa Stop(), Reset()
	 */
	void Start( sTimerID id );

	/**
	 * \fn void Stop( sTimerID id )
	 * \brief Stop the timer
	 * \sa Start(), Reset()
	 */
	void Stop( sTimerID id );

	/**
	 * \fn void AddSample( const sTimerID id, const unsigned long long n )
	 * \brief Explicitely add a timing sample into the class
	 */
	void AddSample( const sTimerID id, const unsigned long long n );

	/**
	 * \fn void Reset(void)
	 * \brief Reset the timer to 0
	 * \sa Start(), Stop()
	 */
	void Clear( );

	/**
	 * \fn void Reset(void)
	 * \brief Reset the timer to 0
	 * \sa Start(), Stop()
	 */
	void Reset( );

	void Reserve( unsigned int nEvents, unsigned int nSamples );

	sTimerID getUniqueID( const std::string& label, unsigned int groupID );

	//	Calculate the average/mean of data for a given event
	void	setNormalize( bool norm );

	//	Calculate the average/mean of data for a given event
	double	getMean( sTimerID id ) const;

	//	Calculate the variance of data for a given event
	//	Variance - average of the squared differences between data points and the mean
	double	getVariance( sTimerID id ) const;

	//	Sqrt of variance, also in units of the original data
	double	getStdDev( sTimerID id ) const;

	/**
	 * \fn double getAverageTime(sTimerID id) const
	 * \return Return the arithmetic mean of all the samples that have been saved
	 */
	double getAverageTime( sTimerID id ) const;

	/**
	 * \fn double getMinimumTime(sTimerID id) const
	 * \return Return the arithmetic min of all the samples that have been saved
	 */
	double getMinimumTime( sTimerID id ) const;

	//	Using the stdDev of the entire population (of an id), eliminate those samples that fall
	//	outside some specified multiple of the stdDev.  This assumes that the population
	//	form a gaussian curve.
	unsigned int	pruneOutliers( double multiple );
	unsigned int	pruneOutliers( sTimerID id , double multiple );
};

#endif // _STATISTICALTIMER_H_
